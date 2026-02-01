"""AGI execution loop with tool use and memory updates."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .model import AGIModel
from ..base.memory import RecurrentState


class AGIExecutor:
    """Execute AGI model with tool use and memory consolidation."""

    def __init__(self, model: AGIModel, max_steps: int = 100) -> None:
        self.model = model
        self.max_steps = max_steps
        self.execution_history: List[Dict[str, Any]] = []

    def execute_step(
        self,
        input_ids: Optional[torch.Tensor],
        obs: Optional[torch.Tensor],
        state: RecurrentState,
        task_ids: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Execute one step with tool use."""
        outputs = self.model(
            input_ids=input_ids,
            obs=obs,
            state=state,
            task_ids=task_ids,
            image=image,
            mode="inference"
        )
        
        if "tool_use" in outputs and outputs["tool_use"]["should_use"]:
            tool_info = outputs["tool_use"]
            tool_id = tool_info["tool_id"]
            
            if isinstance(tool_id, torch.Tensor):
                tool_id = int(tool_id.item())
            
            tool_names = self.model.tool_registry.list_tools()
            if 0 <= tool_id < len(tool_names):
                tool_name = tool_names[tool_id]
                tool_function = self.model.tool_registry.get_tool(tool_name)
                
                if tool_function is not None:
                    try:
                        tool_params = self._extract_tool_params(tool_info)
                        result = tool_function(**tool_params)
                        
                        if hasattr(self.model, "tool_controller") and hasattr(self.model.tool_controller, "tool_executor"):
                            result_tensor = self._result_to_tensor(result, outputs["tool_use"]["context"].device)
                            integrated_result = self.model.tool_controller.tool_executor.integrate_result(
                                result_tensor,
                                outputs["tool_use"]["context"]
                            )
                            outputs["tool_result"] = integrated_result
                        else:
                            outputs["tool_result"] = result
                        
                        outputs["tool_executed"] = True
                        outputs["tool_name"] = tool_name
                    except Exception as e:
                        outputs["tool_error"] = str(e)
                        outputs["tool_executed"] = False
        
        return outputs

    def _extract_tool_params(self, tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from tool info."""
        params = {}
        
        if "params" in tool_info:
            tool_params = tool_info["params"]
            if isinstance(tool_params, torch.Tensor):
                params["x"] = float(tool_params[0, 0].item()) if tool_params.numel() > 0 else 0.0
                params["y"] = float(tool_params[0, 1].item()) if tool_params.numel() > 1 else 0.0
        
        return params

    def _result_to_tensor(self, result: Any, device: torch.device) -> torch.Tensor:
        """Convert tool result to tensor."""
        if isinstance(result, torch.Tensor):
            return result.to(device)
        elif isinstance(result, (int, float)):
            return torch.tensor([result], dtype=torch.float32, device=device).unsqueeze(0)
        elif isinstance(result, dict):
            values = list(result.values())
            if values and isinstance(values[0], (int, float)):
                return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)
        
        return torch.zeros(1, self.model.cfg.hidden_size, device=device)

    def execute_episode(
        self,
        initial_obs: torch.Tensor,
        max_steps: Optional[int] = None,
        task_ids: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """Execute full episode with tool use and memory updates."""
        max_steps = max_steps or self.max_steps
        
        state = self.model.init_state(batch_size=1, device=initial_obs.device)
        obs = initial_obs
        
        episode_history = []
        
        for step in range(max_steps):
            input_ids = torch.zeros(1, 1, dtype=torch.long, device=obs.device)
            
            step_output = self.execute_step(
                input_ids=input_ids,
                obs=obs,
                state=state,
                task_ids=task_ids
            )
            
            episode_history.append({
                "step": step,
                "outputs": step_output,
                "obs": obs.clone(),
            })
            
            if "state" in step_output:
                state = step_output["state"]
            
            if step % self.model.cfg.memory_consolidate_every == 0:
                self._consolidate_memory(episode_history)
            
            if step_output.get("done", False):
                break
            
            action = step_output.get("action")
            if action is not None:
                obs = self._apply_action(obs, action)
        
        self.execution_history.extend(episode_history)
        return episode_history

    def _consolidate_memory(self, history: List[Dict[str, Any]]) -> None:
        """Consolidate episodic memory."""
        if not hasattr(self.model, "hierarchical_memory"):
            return
        
        recent_history = history[-self.model.cfg.episodic_sequence_length:]
        
        if len(recent_history) < 2:
            return
        
        sequence_features = []
        for entry in recent_history:
            if "outputs" in entry and "hidden" in entry["outputs"]:
                hidden = entry["outputs"]["hidden"]
                if hidden.dim() == 3:
                    hidden = hidden[:, -1, :]
                sequence_features.append(hidden.squeeze(0))
        
        if sequence_features:
            sequence_tensor = torch.stack(sequence_features)
            
            with torch.no_grad():
                self.model.hierarchical_memory.episodic_memory.add_episode(sequence_tensor)

    def _apply_action(self, obs: torch.Tensor, action: Any) -> torch.Tensor:
        """Apply action to observation (placeholder)."""
        if isinstance(action, torch.Tensor):
            action_delta = torch.randn_like(obs) * 0.1
            return obs + action_delta
        return obs

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total_steps = len(self.execution_history)
        tools_used = sum(1 for entry in self.execution_history if entry.get("outputs", {}).get("tool_executed", False))
        
        return {
            "total_steps": total_steps,
            "tools_used": tools_used,
            "tool_usage_rate": tools_used / max(total_steps, 1),
        }
    
    def observe_experience(
        self,
        obs: torch.Tensor,
        outputs: Dict[str, Any]
    ) -> None:
        """Observe experience for continuous learning."""
        if not hasattr(self.model, "_continuous_learner") or self.model._continuous_learner is None:
            return
        
        obs_value = obs if obs.dim() == 2 else obs.unsqueeze(0)
        action = outputs.get("action_logits", torch.zeros(1, self.model.cfg.action_dim, device=obs.device)).argmax(dim=-1)
        reward = torch.zeros(obs_value.size(0), device=obs.device)
        value = outputs.get("value", torch.zeros(obs_value.size(0), 1, device=obs.device))
        
        try:
            self.model._continuous_learner.observe(
                state={"obs": obs_value, "value": value},
                action=action,
                reward=reward,
                next_state={"obs": obs_value, "value": value},
                done=False
            )
        except Exception:
            pass
    
    def check_metacognition(
        self,
        task_ids: Optional[torch.Tensor],
        device: torch.device
    ) -> tuple[bool, str, Dict[str, Any]]:
        """Check if model should attempt task via meta-cognition."""
        if not hasattr(self.model, "metacognition") or not self.model.cfg.use_metacognition or task_ids is None:
            return True, "no_metacognition", {}
        
        task_emb = torch.randn(1, self.model.cfg.metacog_task_embedding_dim, device=device)
        
        should_attempt, reason, metrics = self.model.metacognition.should_i_attempt(task_emb)
        
        return should_attempt, reason, metrics
