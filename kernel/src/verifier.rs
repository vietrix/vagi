use std::sync::Mutex;

use anyhow::Result;
use regex::Regex;
use wasmtime::{Config, Engine, Linker, Module, Store};
use wasmtime_wasi::WasiCtxBuilder;
use wasmtime_wasi::p1::{self, WasiP1Ctx};

use crate::hdc::HdcVerifier;
use crate::models::{VerifierActRequest, VerifierActResponse, VerifierRequest, VerifierResponse};

const WASI_SMOKE_WAT: &str = r#"
(module
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
)
"#;

pub struct Verifier {
    engine: Engine,
    smoke_module: Module,
    type_mismatch_regex: Regex,
    /// HDC verifier for parallel pattern-matching checks.
    hdc: Mutex<HdcVerifier>,
}

impl Verifier {
    pub fn new() -> Result<Self> {
        let mut config = Config::new();
        config.consume_fuel(true);
        let engine = Engine::new(&config)?;
        let smoke_module = Module::new(&engine, WASI_SMOKE_WAT)?;
        let type_mismatch_regex = Regex::new(r#"let\s+\w+\s*:\s*u\d+\s*=\s*".*""#)?;
        Ok(Self {
            engine,
            smoke_module,
            type_mismatch_regex,
            hdc: Mutex::new(HdcVerifier::new()),
        })
    }

    pub fn check(&self, request: &VerifierRequest) -> VerifierResponse {
        // Run static checks and HDC checks in parallel.
        let (mut violations, hdc_violations) = std::thread::scope(|s| {
            let static_handle = s.spawn(|| self.static_checks(request));
            let hdc_handle = s.spawn(|| {
                let mut hdc = self.hdc.lock().unwrap_or_else(|e| e.into_inner());
                hdc.check(&request.patch_ir, 0.10)
            });

            let static_v = static_handle.join().unwrap_or_default();
            let hdc_v = hdc_handle.join().unwrap_or_default();
            (static_v, hdc_v)
        });

        // Merge HDC violations.
        violations.extend(hdc_violations);

        let timeout_ms = request.timeout_ms.unwrap_or(50).clamp(1, 2_000);
        let wasi_result = self.run_wasi_smoke(timeout_ms);
        let wasi_ok = wasi_result.is_ok();
        let timeout_hit = wasi_result
            .as_ref()
            .err()
            .map(|e| e.to_string().contains("fuel"))
            .unwrap_or(false);
        if let Err(err) = wasi_result {
            violations.push(format!("wasi_runtime_error:{err}"));
        }

        let act_script = format!(
            "echo verifier_act_probe\nset patch_bytes={}\nappend mode=feedback",
            request.patch_ir.len()
        );
        let act_feedback = self.execute_act(&VerifierActRequest {
            patch_ir: act_script,
            max_steps: Some(12),
            max_output_bytes: Some(4_096),
        });
        if !act_feedback.pass {
            violations.push(format!("wasi_act_exit:{}", act_feedback.exit_code));
        }
        let logic_penalty = self.compute_logic_penalty(&violations, &act_feedback);

        VerifierResponse {
            pass: violations.is_empty(),
            violations,
            cost: request.patch_ir.len() as u32,
            timeout_hit,
            wasi_ok,
            logic_penalty,
        }
    }

    pub fn compute_logic_penalty(
        &self,
        violations: &[String],
        act_feedback: &VerifierActResponse,
    ) -> f32 {
        let mut penalty = 0.0_f32;
        penalty += (violations.len() as f32) * 0.08;
        if !act_feedback.pass {
            penalty += 0.25;
        }
        if !act_feedback.stderr.trim().is_empty() {
            penalty += 0.10;
        }
        if act_feedback.steps_executed >= 12 {
            penalty += 0.05;
        }
        penalty.min(1.0)
    }

    fn static_checks(&self, request: &VerifierRequest) -> Vec<String> {
        let mut violations = Vec::new();
        let ir = request.patch_ir.to_lowercase();
        let max_loop_iters = request.max_loop_iters.unwrap_or(0);
        let side_effect_budget = request.side_effect_budget.unwrap_or(3) as usize;

        if (ir.contains("while(true)") || ir.contains("loop {")) && max_loop_iters == 0 {
            violations.push("infinite_loop_risk".to_string());
        }

        if ir.contains("unsafe") {
            violations.push("unsafe_block_detected".to_string());
        }

        if self.type_mismatch_regex.is_match(&request.patch_ir) {
            violations.push("type_mismatch_literal_assignment".to_string());
        }

        let side_effects = ["write_file", "network_call", "spawn_process", "exec("]
            .iter()
            .filter(|keyword| ir.contains(*keyword))
            .count();
        if side_effects > side_effect_budget {
            violations.push(format!(
                "side_effect_budget_exceeded:{side_effects}>{side_effect_budget}"
            ));
        }

        if ir.len() > 40_000 {
            violations.push("patch_ir_too_large".to_string());
        }

        violations
    }

    fn run_wasi_smoke(&self, timeout_ms: u64) -> Result<()> {
        let mut linker: Linker<WasiP1Ctx> = Linker::new(&self.engine);
        p1::add_to_linker_sync(&mut linker, |wasi_ctx| wasi_ctx)?;

        let mut store = Store::new(&self.engine, WasiCtxBuilder::new().build_p1());
        let fuel = timeout_ms.saturating_mul(1_000);
        store.set_fuel(fuel)?;
        let _ = linker.instantiate(&mut store, &self.smoke_module)?;
        Ok(())
    }

    /// Execute a constrained action script in a WASI-inspired sandbox runtime.
    ///
    /// Accepted commands per line:
    /// - `echo <text>`
    /// - `warn <text>`
    /// - `set <key>=<value>`
    /// - `append <key>=<value>`
    /// - `fail <message>`
    pub fn execute_act(&self, request: &VerifierActRequest) -> VerifierActResponse {
        let max_steps = request.max_steps.unwrap_or(16).clamp(1, 128);
        let max_output_bytes = request.max_output_bytes.unwrap_or(8_192).clamp(256, 65_536);

        let mut stdout_lines: Vec<String> = Vec::new();
        let mut stderr_lines: Vec<String> = Vec::new();
        let mut state_changes: Vec<String> = Vec::new();
        let mut steps_executed = 0_u32;
        let mut exit_code = 0_i32;

        for raw_line in request.patch_ir.lines() {
            if steps_executed >= max_steps {
                stderr_lines.push(format!("step budget exceeded: {max_steps}"));
                exit_code = 124;
                break;
            }
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            steps_executed += 1;

            if let Some(payload) = line.strip_prefix("echo ") {
                stdout_lines.push(payload.trim().to_string());
                continue;
            }
            if let Some(payload) = line.strip_prefix("warn ") {
                stderr_lines.push(payload.trim().to_string());
                continue;
            }
            if let Some(payload) = line.strip_prefix("set ") {
                if let Some((key, value)) = payload.split_once('=') {
                    state_changes.push(format!(
                        "set:{}={}",
                        key.trim(),
                        value.trim()
                    ));
                } else {
                    stderr_lines.push(format!("invalid set command: {line}"));
                    exit_code = 2;
                    break;
                }
                continue;
            }
            if let Some(payload) = line.strip_prefix("append ") {
                if let Some((key, value)) = payload.split_once('=') {
                    state_changes.push(format!(
                        "append:{}={}",
                        key.trim(),
                        value.trim()
                    ));
                } else {
                    stderr_lines.push(format!("invalid append command: {line}"));
                    exit_code = 2;
                    break;
                }
                continue;
            }
            if let Some(payload) = line.strip_prefix("fail ") {
                stderr_lines.push(payload.trim().to_string());
                exit_code = 1;
                break;
            }

            stderr_lines.push(format!("unsupported command: {line}"));
            exit_code = 2;
            break;
        }

        let mut stdout = stdout_lines.join("\n");
        let mut stderr = stderr_lines.join("\n");
        if stdout.len() > max_output_bytes {
            stdout.truncate(max_output_bytes);
        }
        if stderr.len() > max_output_bytes {
            stderr.truncate(max_output_bytes);
        }

        VerifierActResponse {
            pass: exit_code == 0,
            exit_code,
            stdout,
            stderr,
            state_changes,
            steps_executed,
            runtime: "wasi-act-v1".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::models::{VerifierActRequest, VerifierRequest};

    use super::Verifier;

    #[test]
    fn catches_infinite_loop_without_budget() {
        let verifier = Verifier::new().expect("create verifier");
        let res = verifier.check(&VerifierRequest {
            patch_ir: "while(true){ do_work(); }".to_string(),
            max_loop_iters: None,
            side_effect_budget: Some(3),
            timeout_ms: Some(10),
        });
        assert!(!res.pass);
        assert!(res
            .violations
            .iter()
            .any(|entry| entry.contains("infinite_loop_risk")));
    }

    #[test]
    fn allows_simple_safe_ir() {
        let verifier = Verifier::new().expect("create verifier");
        let res = verifier.check(&VerifierRequest {
            patch_ir: "validate(input); hash(password);".to_string(),
            max_loop_iters: Some(128),
            side_effect_budget: Some(3),
            timeout_ms: Some(10),
        });
        assert!(res.wasi_ok);
        assert!(res.pass);
    }

    #[test]
    fn executes_wasi_act_script() {
        let verifier = Verifier::new().expect("create verifier");
        let res = verifier.execute_act(&VerifierActRequest {
            patch_ir: "echo started\nset phase=observe\nappend log=done".to_string(),
            max_steps: Some(8),
            max_output_bytes: Some(512),
        });
        assert!(res.pass);
        assert_eq!(res.exit_code, 0);
        assert!(res.stdout.contains("started"));
        assert!(res.state_changes.iter().any(|e| e.contains("phase")));
    }
}
