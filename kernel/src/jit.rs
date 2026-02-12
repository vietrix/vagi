use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use wasmtime::{Engine, Instance, Module, Store};

use crate::models::{JitExecuteRequest, JitExecuteResponse};

const MAX_LOGIC_OPS: usize = 256;

#[derive(Debug, Clone, Copy)]
enum LogicOp {
    Add(i64),
    Sub(i64),
    Mul(i64),
    Xor(i64),
    And(i64),
    Or(i64),
    Shl(u32),
    Shr(u32),
}

impl LogicOp {
    fn parse(line: &str) -> Result<Self> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            bail!("logic contains an empty operation line");
        }
        let mut parts = trimmed.split_whitespace();
        let op = parts
            .next()
            .ok_or_else(|| anyhow!("invalid operation line: `{trimmed}`"))?
            .to_ascii_lowercase();
        let raw_value = parts
            .next()
            .ok_or_else(|| anyhow!("operation `{op}` requires one numeric argument"))?;
        if parts.next().is_some() {
            bail!("operation `{op}` only accepts one argument");
        }

        let value_i64: i64 = raw_value
            .parse()
            .map_err(|_| anyhow!("invalid numeric argument `{raw_value}`"))?;

        match op.as_str() {
            "add" => Ok(Self::Add(value_i64)),
            "sub" => Ok(Self::Sub(value_i64)),
            "mul" => Ok(Self::Mul(value_i64)),
            "xor" => Ok(Self::Xor(value_i64)),
            "and" => Ok(Self::And(value_i64)),
            "or" => Ok(Self::Or(value_i64)),
            "shl" => {
                let shift: u32 = value_i64
                    .try_into()
                    .map_err(|_| anyhow!("shift must be non-negative"))?;
                Ok(Self::Shl(shift))
            }
            "shr" => {
                let shift: u32 = value_i64
                    .try_into()
                    .map_err(|_| anyhow!("shift must be non-negative"))?;
                Ok(Self::Shr(shift))
            }
            _ => bail!("unsupported op `{op}`"),
        }
    }

    fn normalized(self) -> String {
        match self {
            Self::Add(v) => format!("add {v}"),
            Self::Sub(v) => format!("sub {v}"),
            Self::Mul(v) => format!("mul {v}"),
            Self::Xor(v) => format!("xor {v}"),
            Self::And(v) => format!("and {v}"),
            Self::Or(v) => format!("or {v}"),
            Self::Shl(v) => format!("shl {v}"),
            Self::Shr(v) => format!("shr {v}"),
        }
    }

    fn emit_wat(self) -> String {
        match self {
            Self::Add(v) => format!("i64.const {v}\n    i64.add"),
            Self::Sub(v) => format!("i64.const {v}\n    i64.sub"),
            Self::Mul(v) => format!("i64.const {v}\n    i64.mul"),
            Self::Xor(v) => format!("i64.const {v}\n    i64.xor"),
            Self::And(v) => format!("i64.const {v}\n    i64.and"),
            Self::Or(v) => format!("i64.const {v}\n    i64.or"),
            Self::Shl(v) => format!("i64.const {v}\n    i64.shl"),
            Self::Shr(v) => format!("i64.const {v}\n    i64.shr_u"),
        }
    }
}

pub struct JitEngine {
    engine: Engine,
}

impl JitEngine {
    pub fn new() -> Result<Self> {
        Ok(Self {
            engine: Engine::default(),
        })
    }

    pub fn compile_and_execute(&self, request: &JitExecuteRequest) -> Result<JitExecuteResponse> {
        let ops = parse_logic(&request.logic)?;
        let normalized_logic = ops.iter().map(|op| op.normalized()).collect::<Vec<_>>();
        let wat = build_wat(&ops);

        let compile_started = Instant::now();
        let module = Module::new(&self.engine, wat)?;
        let compile_micros = elapsed_micros_u64(compile_started.elapsed());

        let execute_started = Instant::now();
        let mut store = Store::new(&self.engine, ());
        let instance = Instance::new(&mut store, &module, &[])?;
        let function = instance
            .get_typed_func::<i64, i64>(&mut store, "reasoning_path")
            .map_err(|err| anyhow!("missing exported function `reasoning_path`: {err}"))?;
        let output = function.call(&mut store, request.input)?;
        let execute_micros = elapsed_micros_u64(execute_started.elapsed());

        Ok(JitExecuteResponse {
            output,
            backend: "wasmtime-cranelift-jit",
            op_count: ops.len(),
            compile_micros,
            execute_micros,
            normalized_logic,
        })
    }
}

fn parse_logic(logic: &str) -> Result<Vec<LogicOp>> {
    let mut ops = Vec::new();
    for line in logic.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        ops.push(LogicOp::parse(trimmed)?);
        if ops.len() > MAX_LOGIC_OPS {
            bail!("logic exceeds max operation count: {MAX_LOGIC_OPS}");
        }
    }
    if ops.is_empty() {
        bail!("logic is empty");
    }
    Ok(ops)
}

fn build_wat(ops: &[LogicOp]) -> String {
    let instructions = ops
        .iter()
        .map(|op| op.emit_wat())
        .collect::<Vec<_>>()
        .join("\n    ");
    format!(
        "(module
  (func (export \"reasoning_path\") (param i64) (result i64)
    local.get 0
    {instructions}
  )
)"
    )
}

fn elapsed_micros_u64(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{JitEngine, parse_logic};
    use crate::models::JitExecuteRequest;

    #[test]
    fn parse_logic_ignores_empty_and_comment_lines() {
        let logic = "# comment\nadd 4\n\nmul 3\n";
        let ops = parse_logic(logic).expect("parse logic");
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn jit_execute_runs_pipeline() {
        let engine = JitEngine::new().expect("create jit engine");
        let request = JitExecuteRequest {
            logic: "add 5\nmul 2\nxor 3".to_string(),
            input: 7,
        };
        let response = engine
            .compile_and_execute(&request)
            .expect("compile and execute");

        let expected = 7i64.wrapping_add(5).wrapping_mul(2) ^ 3;
        assert_eq!(response.output, expected);
        assert_eq!(response.op_count, 3);
        assert_eq!(response.backend, "wasmtime-cranelift-jit");
    }
}
