"""
Monte Carlo Tree Search (MCTS) Engine for System 2 Reasoning.

This module implements MCTS to guide language model generation through
explicit exploration of reasoning paths before committing to answers.

Mathematical Foundations:

MCTS builds a search tree where:
- Nodes represent states (partial reasoning chains)
- Edges represent actions (next thoughts/tokens)

Four phases per simulation:
1. Selection: Navigate tree using UCT until reaching unexpanded node
2. Expansion: Generate new child nodes (possible next thoughts)
3. Evaluation: Estimate value of new nodes using value model
4. Backpropagation: Update values up the tree

UCT (Upper Confidence Bound for Trees):
    UCT(s, a) = Q(s, a) + c * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

    where:
    - Q(s, a): Average value of taking action a from state s
    - c: Exploration constant (sqrt(2) theoretically optimal)
    - P(s, a): Prior probability from policy model
    - N(s): Visit count of parent
    - N(s, a): Visit count of this action

This enables:
- Exploring multiple reasoning paths before committing
- Self-verification of logical steps
- Backtracking from dead ends
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any, Tuple
from abc import ABC, abstractmethod
import heapq
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MCTSConfig:
    """Configuration for MCTS reasoning engine."""
    num_simulations: int = 50
    max_depth: int = 10
    c_puct: float = 1.414  # Exploration constant
    num_expansions: int = 5
    temperature: float = 0.8
    value_weight: float = 0.5
    discount_factor: float = 0.99
    virtual_loss: float = 1.0
    dirichlet_alpha: float = 0.3  # For exploration at root
    dirichlet_noise_weight: float = 0.25


class MCTSNode:
    """
    Node in the MCTS search tree.

    Each node represents a state in the reasoning chain, storing:
    - The text/thought at this node
    - Statistics for UCT selection
    - Links to parent and children
    """

    def __init__(
        self,
        state: str,
        parent: Optional["MCTSNode"] = None,
        prior: float = 1.0,
        action: Optional[str] = None,
    ):
        """
        Initialize MCTS node.

        Args:
            state: The current reasoning state (accumulated text)
            parent: Parent node (None for root)
            prior: Prior probability from policy model P(a|s)
            action: The action (thought) that led to this state
        """
        self.state = state
        self.parent = parent
        self.prior = prior
        self.action = action

        # Tree structure
        self.children: Dict[str, "MCTSNode"] = {}

        # Statistics
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.virtual_loss: float = 0.0  # For parallel MCTS

        # Flags
        self.is_terminal: bool = False
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        """Average value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def effective_visit_count(self) -> float:
        """Visit count including virtual loss."""
        return self.visit_count + self.virtual_loss

    def uct_score(self, c_puct: float, parent_visits: int) -> float:
        """
        Compute UCT score for selection.

        UCT = Q + c * P * sqrt(N_parent) / (1 + N)

        Higher score = more promising to explore.
        """
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.effective_visit_count)
        return self.q_value + exploration

    def select_child(self, c_puct: float) -> "MCTSNode":
        """Select child with highest UCT score."""
        return max(
            self.children.values(),
            key=lambda child: child.uct_score(c_puct, self.visit_count)
        )

    def expand(
        self,
        actions: List[str],
        priors: List[float],
        next_states: List[str],
    ) -> List["MCTSNode"]:
        """
        Expand node with new children.

        Args:
            actions: Possible next thoughts/actions
            priors: Prior probabilities for each action
            next_states: Resulting states after each action

        Returns:
            List of new child nodes
        """
        new_children = []

        for action, prior, next_state in zip(actions, priors, next_states):
            if action not in self.children:
                child = MCTSNode(
                    state=next_state,
                    parent=self,
                    prior=prior,
                    action=action,
                )
                self.children[action] = child
                new_children.append(child)

        self.is_expanded = True
        return new_children

    def backpropagate(self, value: float, discount: float = 1.0):
        """
        Backpropagate value up the tree.

        Updates value_sum and visit_count for this node
        and all ancestors.
        """
        node = self
        current_value = value

        while node is not None:
            node.visit_count += 1
            node.value_sum += current_value
            node.virtual_loss = max(0, node.virtual_loss - 1)

            current_value *= discount
            node = node.parent

    def add_virtual_loss(self, amount: float = 1.0):
        """Add virtual loss for parallel MCTS."""
        node = self
        while node is not None:
            node.virtual_loss += amount
            node = node.parent

    def get_path(self) -> List[str]:
        """Get sequence of actions from root to this node."""
        path = []
        node = self
        while node.parent is not None:
            path.append(node.action)
            node = node.parent
        return list(reversed(path))

    def to_dict(self) -> dict:
        """Serialize node for debugging."""
        return {
            "state": self.state[:100] + "..." if len(self.state) > 100 else self.state,
            "action": self.action,
            "visits": self.visit_count,
            "value": self.q_value,
            "prior": self.prior,
            "children": list(self.children.keys()),
        }


class PolicyModel(ABC):
    """Abstract interface for the policy model (LLM)."""

    @abstractmethod
    def generate_thoughts(
        self,
        state: str,
        num_thoughts: int,
        temperature: float,
    ) -> Tuple[List[str], List[float]]:
        """
        Generate possible next thoughts/actions.

        Args:
            state: Current reasoning state
            num_thoughts: Number of thoughts to generate
            temperature: Sampling temperature

        Returns:
            (thoughts, priors): Lists of thought strings and prior probabilities
        """
        pass

    @abstractmethod
    def get_next_state(self, state: str, thought: str) -> str:
        """Combine state and thought into new state."""
        pass


class ValueModel(ABC):
    """Abstract interface for the value/verifier model."""

    @abstractmethod
    def evaluate(self, state: str) -> float:
        """
        Evaluate the quality of a reasoning state.

        Args:
            state: Current reasoning state

        Returns:
            Value in [0, 1] indicating quality/correctness
        """
        pass

    @abstractmethod
    def is_terminal(self, state: str) -> bool:
        """Check if state is a terminal/complete reasoning chain."""
        pass


class LLMPolicyModel(PolicyModel):
    """
    Policy model using a language model for thought generation.

    Wraps a causal LM to generate candidate reasoning steps.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        thought_prefix: str = "\nStep: ",
        thought_suffix: str = "\n",
        max_thought_length: int = 100,
    ):
        """
        Initialize LLM policy.

        Args:
            model: The language model (VAGIForCausalLM or similar)
            tokenizer: Tokenizer for encoding/decoding
            thought_prefix: Prefix for each reasoning step
            thought_suffix: Suffix indicating end of step
            max_thought_length: Maximum tokens per thought
        """
        self.model = model
        self.tokenizer = tokenizer
        self.thought_prefix = thought_prefix
        self.thought_suffix = thought_suffix
        self.max_thought_length = max_thought_length

    def generate_thoughts(
        self,
        state: str,
        num_thoughts: int,
        temperature: float,
    ) -> Tuple[List[str], List[float]]:
        """Generate diverse candidate thoughts using sampling."""
        thoughts = []
        priors = []

        # Prepare input
        prompt = state + self.thought_prefix
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(next(self.model.parameters()).device)

        # Generate multiple samples
        with torch.no_grad():
            for _ in range(num_thoughts):
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_thought_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    use_cache=True,
                )

                # Decode generated text
                generated = self.tokenizer.decode(
                    outputs[0][input_ids.shape[1]:],
                    skip_special_tokens=True,
                )

                # Extract thought (up to suffix)
                if self.thought_suffix in generated:
                    thought = generated[:generated.index(self.thought_suffix)]
                else:
                    thought = generated

                thought = thought.strip()
                if thought and thought not in thoughts:
                    thoughts.append(thought)

                    # Estimate prior from perplexity
                    prior = self._estimate_prior(prompt, thought)
                    priors.append(prior)

        # Normalize priors
        if priors:
            total = sum(priors)
            priors = [p / total for p in priors]
        else:
            # Fallback: uniform priors
            priors = [1.0 / num_thoughts] * len(thoughts)

        return thoughts, priors

    def _estimate_prior(self, prompt: str, thought: str) -> float:
        """Estimate prior probability using model likelihood."""
        full_text = prompt + thought
        input_ids = self.tokenizer.encode(full_text, return_tensors="pt")
        input_ids = input_ids.to(next(self.model.parameters()).device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs["loss"]

        # Convert loss to probability-like score
        return math.exp(-loss.item())

    def get_next_state(self, state: str, thought: str) -> str:
        """Append thought to state."""
        return state + self.thought_prefix + thought + self.thought_suffix


class VerifierValueModel(ValueModel):
    """
    Value model using a trained verifier/reward model.

    Evaluates reasoning chains for correctness and quality.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        terminal_tokens: List[str] = None,
    ):
        """
        Initialize verifier.

        Args:
            model: Reward/verifier model
            tokenizer: Tokenizer
            terminal_tokens: Tokens indicating reasoning completion
        """
        self.model = model
        self.tokenizer = tokenizer
        self.terminal_tokens = terminal_tokens or ["Answer:", "Therefore:", "Final answer:"]

    def evaluate(self, state: str) -> float:
        """Evaluate state using verifier model."""
        input_ids = self.tokenizer.encode(state, return_tensors="pt")
        input_ids = input_ids.to(next(self.model.parameters()).device)

        with torch.no_grad():
            # Assume model outputs reward/value
            outputs = self.model(input_ids)
            if isinstance(outputs, dict) and "value" in outputs:
                value = outputs["value"]
            elif isinstance(outputs, dict) and "logits" in outputs:
                # Use last token logit as value proxy
                value = torch.sigmoid(outputs["logits"][:, -1].mean())
            else:
                value = outputs

        return float(value.squeeze())

    def is_terminal(self, state: str) -> bool:
        """Check for terminal tokens in state."""
        return any(token in state for token in self.terminal_tokens)


class SelfEvalValueModel(ValueModel):
    """
    Value model using self-evaluation prompting.

    Uses the same LLM to evaluate its own reasoning quality.
    Useful when a dedicated verifier isn't available.
    """

    EVAL_PROMPT = """
Evaluate the following reasoning chain on a scale from 0 to 10.
Consider: logical consistency, factual accuracy, and progress toward answer.

Reasoning:
{state}

Evaluation (just the number):"""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        terminal_tokens: List[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.terminal_tokens = terminal_tokens or ["Answer:", "Therefore:", "Final answer:"]

    def evaluate(self, state: str) -> float:
        """Evaluate using self-prompting."""
        prompt = self.EVAL_PROMPT.format(state=state)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(next(self.model.parameters()).device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False,
            )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Parse numeric response
        try:
            score = float(response.split()[0])
            return min(max(score / 10.0, 0.0), 1.0)  # Normalize to [0, 1]
        except (ValueError, IndexError):
            return 0.5  # Default middle value

    def is_terminal(self, state: str) -> bool:
        return any(token in state for token in self.terminal_tokens)


class MCTSEngine:
    """
    Monte Carlo Tree Search Engine for Reasoning.

    Orchestrates the MCTS process to guide language model generation
    through explicit exploration of reasoning paths.
    """

    def __init__(
        self,
        policy_model: PolicyModel,
        value_model: ValueModel,
        config: MCTSConfig = None,
    ):
        """
        Initialize MCTS engine.

        Args:
            policy_model: Model for generating candidate thoughts
            value_model: Model for evaluating reasoning states
            config: MCTS hyperparameters
        """
        self.policy = policy_model
        self.value = value_model
        self.config = config or MCTSConfig()

    def search(
        self,
        initial_state: str,
        num_simulations: Optional[int] = None,
    ) -> MCTSNode:
        """
        Run MCTS from initial state.

        Args:
            initial_state: Starting reasoning state (e.g., question)
            num_simulations: Override config num_simulations

        Returns:
            Root node of the search tree
        """
        num_sims = num_simulations or self.config.num_simulations

        # Initialize root
        root = MCTSNode(state=initial_state, parent=None)

        # Add Dirichlet noise at root for exploration
        self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(num_sims):
            self._simulate(root)

        return root

    def _simulate(self, root: MCTSNode):
        """Run a single MCTS simulation."""
        node = root

        # ============ Phase 1: Selection ============
        # Navigate tree using UCT until we reach an unexpanded node
        while node.is_expanded and node.children and not node.is_terminal:
            node = node.select_child(self.config.c_puct)

        # ============ Phase 2: Expansion ============
        if not node.is_terminal and not node.is_expanded:
            # Check depth limit
            depth = len(node.get_path())
            if depth < self.config.max_depth:
                # Generate candidate thoughts
                thoughts, priors = self.policy.generate_thoughts(
                    node.state,
                    self.config.num_expansions,
                    self.config.temperature,
                )

                if thoughts:
                    # Create child states
                    next_states = [
                        self.policy.get_next_state(node.state, thought)
                        for thought in thoughts
                    ]

                    # Expand node
                    new_children = node.expand(thoughts, priors, next_states)

                    # Check for terminal states
                    for child in new_children:
                        child.is_terminal = self.value.is_terminal(child.state)

                    # Select a new child for evaluation
                    if new_children:
                        node = random.choice(new_children)

        # ============ Phase 3: Evaluation ============
        value = self.value.evaluate(node.state)

        # Boost value for terminal (complete) states
        if node.is_terminal:
            value = max(value, 0.8)  # Terminal states get minimum 0.8

        # ============ Phase 4: Backpropagation ============
        node.backpropagate(value, self.config.discount_factor)

    def _add_dirichlet_noise(self, node: MCTSNode):
        """Add Dirichlet noise to root node priors for exploration."""
        if not node.children:
            # First expand root
            thoughts, priors = self.policy.generate_thoughts(
                node.state,
                self.config.num_expansions,
                self.config.temperature,
            )

            if thoughts:
                next_states = [
                    self.policy.get_next_state(node.state, thought)
                    for thought in thoughts
                ]
                node.expand(thoughts, priors, next_states)

        # Add Dirichlet noise to priors
        if node.children:
            alpha = self.config.dirichlet_alpha
            noise_weight = self.config.dirichlet_noise_weight

            noise = torch.distributions.Dirichlet(
                torch.full((len(node.children),), alpha)
            ).sample().tolist()

            for (child, noise_val) in zip(node.children.values(), noise):
                child.prior = (1 - noise_weight) * child.prior + noise_weight * noise_val

    def get_best_path(self, root: MCTSNode) -> List[str]:
        """
        Get the best reasoning path from search tree.

        Uses visit counts to determine best path (more robust than Q-values).
        """
        path = []
        node = root

        while node.children:
            # Select child with most visits (not highest Q-value)
            best_child = max(
                node.children.values(),
                key=lambda c: c.visit_count
            )
            path.append(best_child.action)
            node = best_child

        return path

    def get_best_state(self, root: MCTSNode) -> str:
        """Get the final state of the best reasoning path."""
        node = root

        while node.children:
            best_child = max(
                node.children.values(),
                key=lambda c: c.visit_count
            )
            node = best_child

        return node.state

    def get_action_probs(
        self,
        root: MCTSNode,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """
        Get action probabilities from visit counts.

        Args:
            root: Root node of search tree
            temperature: Temperature for converting counts to probs

        Returns:
            Dictionary mapping actions to probabilities
        """
        if not root.children:
            return {}

        visits = {
            action: child.visit_count
            for action, child in root.children.items()
        }

        if temperature == 0:
            # Deterministic: select best
            best_action = max(visits, key=visits.get)
            return {action: 1.0 if action == best_action else 0.0 for action in visits}

        # Apply temperature
        total = sum(v ** (1.0 / temperature) for v in visits.values())
        return {
            action: (count ** (1.0 / temperature)) / total
            for action, count in visits.items()
        }

    def think(
        self,
        question: str,
        num_simulations: Optional[int] = None,
        verbose: bool = False,
    ) -> str:
        """
        High-level interface for reasoning about a question.

        Args:
            question: The question/problem to reason about
            num_simulations: Number of MCTS simulations
            verbose: Whether to print search progress

        Returns:
            Complete reasoning chain with answer
        """
        if verbose:
            print(f"Starting MCTS reasoning...")
            print(f"Question: {question[:100]}...")

        # Run search
        root = self.search(question, num_simulations)

        if verbose:
            print(f"Search complete. Tree statistics:")
            print(f"  Root visits: {root.visit_count}")
            print(f"  Number of children: {len(root.children)}")

            # Show top paths
            print("Top reasoning paths:")
            for action, child in sorted(
                root.children.items(),
                key=lambda x: x[1].visit_count,
                reverse=True
            )[:3]:
                print(f"  [{child.visit_count} visits] {action[:50]}...")

        # Get best result
        result = self.get_best_state(root)

        return result


class BeamSearchMCTS(MCTSEngine):
    """
    Beam Search variant of MCTS.

    Maintains a beam of top-k nodes at each depth level,
    combining the efficiency of beam search with MCTS's
    exploration capabilities.
    """

    def __init__(
        self,
        policy_model: PolicyModel,
        value_model: ValueModel,
        config: MCTSConfig = None,
        beam_width: int = 5,
    ):
        super().__init__(policy_model, value_model, config)
        self.beam_width = beam_width

    def search(
        self,
        initial_state: str,
        num_simulations: Optional[int] = None,
    ) -> MCTSNode:
        """Run beam search MCTS."""
        root = MCTSNode(state=initial_state, parent=None)

        # Current beam
        beam = [root]

        for depth in range(self.config.max_depth):
            candidates = []

            for node in beam:
                if node.is_terminal:
                    candidates.append((node, node.q_value))
                    continue

                # Expand node
                thoughts, priors = self.policy.generate_thoughts(
                    node.state,
                    self.config.num_expansions,
                    self.config.temperature,
                )

                if thoughts:
                    next_states = [
                        self.policy.get_next_state(node.state, thought)
                        for thought in thoughts
                    ]

                    children = node.expand(thoughts, priors, next_states)

                    for child in children:
                        child.is_terminal = self.value.is_terminal(child.state)
                        value = self.value.evaluate(child.state)
                        child.backpropagate(value)
                        candidates.append((child, value))

            # Select top-k candidates for next beam
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = [c[0] for c in candidates[:self.beam_width]]

            # Early termination if best is terminal
            if beam and beam[0].is_terminal:
                break

        return root


class ParallelMCTS(MCTSEngine):
    """
    Parallel MCTS with virtual loss.

    Runs multiple simulations in parallel using virtual loss
    to encourage exploration of different paths.
    """

    def search(
        self,
        initial_state: str,
        num_simulations: Optional[int] = None,
        num_workers: int = 4,
    ) -> MCTSNode:
        """Run parallel MCTS simulations."""
        num_sims = num_simulations or self.config.num_simulations

        root = MCTSNode(state=initial_state, parent=None)
        self._add_dirichlet_noise(root)

        # Simulate in batches
        batch_size = num_workers

        for batch_start in range(0, num_sims, batch_size):
            batch_end = min(batch_start + batch_size, num_sims)
            batch_nodes = []

            # Selection phase for all workers
            for _ in range(batch_end - batch_start):
                node = self._select_with_virtual_loss(root)
                batch_nodes.append(node)

            # Expansion and evaluation (can be batched for efficiency)
            values = []
            for node in batch_nodes:
                if not node.is_terminal and not node.is_expanded:
                    self._expand_node(node)

                value = self.value.evaluate(node.state)
                values.append(value)

            # Backpropagation
            for node, value in zip(batch_nodes, values):
                node.backpropagate(value, self.config.discount_factor)

        return root

    def _select_with_virtual_loss(self, root: MCTSNode) -> MCTSNode:
        """Select node while adding virtual loss."""
        node = root

        while node.is_expanded and node.children and not node.is_terminal:
            node.add_virtual_loss(self.config.virtual_loss)
            node = node.select_child(self.config.c_puct)

        node.add_virtual_loss(self.config.virtual_loss)
        return node

    def _expand_node(self, node: MCTSNode):
        """Expand a single node."""
        depth = len(node.get_path())
        if depth >= self.config.max_depth:
            return

        thoughts, priors = self.policy.generate_thoughts(
            node.state,
            self.config.num_expansions,
            self.config.temperature,
        )

        if thoughts:
            next_states = [
                self.policy.get_next_state(node.state, thought)
                for thought in thoughts
            ]
            children = node.expand(thoughts, priors, next_states)

            for child in children:
                child.is_terminal = self.value.is_terminal(child.state)
