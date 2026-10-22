"""All 22 node types as Pydantic v2 models with discriminated union."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

# ---------- Helper models ----------


class ConditionTarget(BaseModel):
    """A condition target for achievements, end conditions, unlock gates."""

    node_id: str
    property: str
    threshold: float


class ConverterIO(BaseModel):
    """Input or output spec for a Converter node."""

    resource: str
    amount: float
    formula: str | None = None


class CostEntry(BaseModel):
    """One currency component of a price: ``{currency, amount}``.

    For generators ``amount`` is the price of the first unit; unit k costs
    ``amount * growth_rate**(k-1)`` where ``growth_rate`` defaults to the
    generator's ``cost_growth_rate``.
    """

    currency: str
    amount: float = Field(ge=0)
    growth_rate: float | None = Field(default=None, gt=0)


class Requirement(BaseModel):
    """Purchase threshold: balance of ``currency`` must be >= ``amount``.

    Checked at every purchase, never consumed.
    """

    currency: str
    amount: float = Field(ge=0)


def _split_cost_list(data, float_field: str):
    """Accept ``cost`` as a float (legacy) or a list of CostEntry dicts."""
    if isinstance(data, dict):
        c = data.get("cost")
        if isinstance(c, list):
            data = dict(data)
            data["costs"] = c
            del data["cost"]
            first = c[0] if c else None
            amt = (first.get("amount") if isinstance(first, dict) else getattr(first, "amount", 0)) if first is not None else 0.0
            data.setdefault(float_field, amt)
    return data


# ---------- NodeBase ----------


class NodeBase(BaseModel):
    """Base fields shared by all node types."""

    id: str
    tags: list[str] = Field(default_factory=list)
    activation_mode: Literal["automatic", "interactive", "passive", "toggle"] = "automatic"
    pull_mode: Literal["pull_any", "pull_all"] = "pull_any"
    cooldown_time: float | None = None


# ---------- 22 Node types ----------


class Resource(NodeBase):
    type: Literal["resource"] = "resource"
    name: str
    initial_value: float = 0.0
    capacity: float | None = None
    overflow_behavior: Literal["clamp", "waste"] = "clamp"


class Generator(NodeBase):
    type: Literal["generator"] = "generator"
    name: str
    base_production: float = Field(gt=0)
    cost_base: float = Field(gt=0)
    cost_growth_rate: float = Field(gt=0)
    cycle_time: float = Field(default=1.0, gt=0)
    # Multi-currency price (input key ``cost`` as a list). None = legacy single
    # price ``cost_base`` in the generator's own currency.
    costs: list[CostEntry] | None = None
    requires: list[Requirement] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _cost_list(cls, data):
        return _split_cost_list(data, "cost_base")


class NestedGenerator(NodeBase):
    type: Literal["nested_generator"] = "nested_generator"
    name: str
    target_generator: str
    production_rate: float
    cost_base: float
    cost_growth_rate: float


class UpgradeEffect(BaseModel):
    """Extra per-generator bonus carried by one upgrade (v0.1.12 multi-effect).

    Lets a single purchase boost several generators by different amounts,
    e.g. one skill level that adds Stone *and* Silver income. Stacks with the
    same rules as the upgrade's own ``target``/``magnitude``.
    """

    target: str
    magnitude: float
    stacking_group: str | None = None  # None = the upgrade's stacking_group


class Upgrade(NodeBase):
    type: Literal["upgrade"] = "upgrade"
    name: str
    upgrade_type: Literal["multiplicative", "additive", "percentage"]
    magnitude: float
    cost: float
    target: str
    stacking_group: str
    duration: float | None = None
    cooldown_time: float | None = None
    # Multi-currency price (input key ``cost`` as a list). None = legacy single
    # price ``cost`` in the primary currency.
    costs: list[CostEntry] | None = None
    requires: list[Requirement] = Field(default_factory=list)
    # v0.1.9 multiplier scope: with target "_all", restrict to generators that
    # produce into one of these resources or carry one of these tags.
    # None = global (backward compatible).
    targets: list[str] | None = None
    # v0.1.12: additional (target, magnitude) bonuses applied on purchase.
    effects: list[UpgradeEffect] | None = None
    # v0.1.12 unlock rules enforced by the engine: purchasable only when every
    # id in prereq_all is purchased AND (prereq_any empty or any purchased).
    prereq_all: list[str] = Field(default_factory=list)
    prereq_any: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _cost_list(cls, data):
        return _split_cost_list(data, "cost")


class PrestigeLayer(NodeBase):
    type: Literal["prestige_layer"] = "prestige_layer"
    name: str = ""
    formula_expr: str
    layer_index: int
    reset_scope: list[str]
    persistence_scope: list[str] = Field(default_factory=list)
    bonus_type: Literal["multiplicative", "additive", "percentage"] = "multiplicative"
    milestone_rules: list[dict] = Field(default_factory=list)
    currency_id: str | None = None
    parent_layer: str | None = None


class SacrificeNode(NodeBase):
    type: Literal["sacrifice"] = "sacrifice"
    name: str = ""
    formula_expr: str
    reset_scope: list[str]
    bonus_type: Literal["multiplicative", "additive", "percentage"] = "multiplicative"


class Achievement(NodeBase):
    type: Literal["achievement"] = "achievement"
    name: str = ""
    condition_type: Literal["single_threshold", "multi_threshold", "collection", "compound"]
    targets: list[ConditionTarget]
    logic: str = "and"
    bonus: dict | None = None
    permanent: bool = True


class Manager(NodeBase):
    type: Literal["manager"] = "manager"
    name: str = ""
    target: str
    automation_type: Literal["collect", "buy", "activate"] = "collect"


class Converter(NodeBase):
    type: Literal["converter"] = "converter"
    name: str = ""
    inputs: list[ConverterIO]
    outputs: list[ConverterIO]
    rate: float = 1.0
    pull_mode: Literal["pull_any", "pull_all"] = "pull_any"
    recipe_type: Literal["fixed", "scaling"] = "fixed"
    conversion_limit: int | None = None


class ProbabilityNode(NodeBase):
    type: Literal["probability"] = "probability"
    name: str = ""
    expected_value: float
    variance: float = Field(default=0.0, ge=0)
    crit_chance: float = Field(default=0.0, ge=0, le=1)
    crit_multiplier: float = 1.0


class EndCondition(NodeBase):
    type: Literal["end_condition"] = "end_condition"
    name: str = ""
    condition_type: Literal["single_threshold", "multi_threshold", "collection", "compound"] = (
        "single_threshold"
    )
    targets: list[ConditionTarget]
    logic: str = "and"


class UnlockGate(NodeBase):
    type: Literal["unlock_gate"] = "unlock_gate"
    name: str = ""
    condition_type: Literal["single_threshold", "multi_threshold", "collection", "compound"] = (
        "single_threshold"
    )
    targets: list[ConditionTarget]
    prerequisites: list[str]
    logic: str = "and"
    permanent: bool = True


class ChoiceGroup(NodeBase):
    type: Literal["choice_group"] = "choice_group"
    name: str = ""
    options: list[str]
    max_selections: int = Field(default=1, ge=1)
    respeccable: bool = False
    respec_cost: float | None = None


class Register(NodeBase):
    type: Literal["register"] = "register"
    name: str = ""
    formula_expr: str
    input_labels: list[dict[str, str]]


class Gate(NodeBase):
    type: Literal["gate"] = "gate"
    name: str = ""
    mode: Literal["deterministic", "probabilistic"]
    weights: list[float] | None = None
    probabilities: list[float] | None = None


class Queue(NodeBase):
    type: Literal["queue"] = "queue"
    name: str = ""
    delay: float = Field(gt=0)
    capacity: int | None = None


class TickspeedNode(NodeBase):
    type: Literal["tickspeed"] = "tickspeed"
    name: str = "Tickspeed"
    base_tickspeed: float = 1.0


class AutobuyerNode(NodeBase):
    type: Literal["autobuyer"] = "autobuyer"
    name: str = ""
    target: str
    interval: float = 1.0
    priority: int = 0
    condition: str | None = None
    bulk_amount: Literal["1", "10", "max"] = "1"
    enabled: bool = True


class DrainNode(NodeBase):
    type: Literal["drain"] = "drain"
    name: str = ""
    rate: float
    condition: str | None = None


class BuffNode(NodeBase):
    type: Literal["buff"] = "buff"
    name: str = ""
    buff_type: Literal["timed", "proc"]
    duration: float | None = None
    proc_chance: float | None = None
    multiplier: float = 2.0
    target: str | None = None
    cooldown: float = 0.0


class SynergyNode(NodeBase):
    type: Literal["synergy"] = "synergy"
    name: str = ""
    sources: list[str]
    formula_expr: str
    target: str


# ---------- Discriminated Union ----------

class FarmStage(BaseModel):
    required_power: float = Field(ge=0)
    drop_per_clear: float = Field(ge=0)


class FarmNode(NodeBase):
    """Stage farming, serial model (v0.1.6).

    One stage is farmed at a time: the highest stage whose ``required_power``
    is <= the current balance of ``power_resource`` (stably clearable; a
    combat-gate win rate replaces this test later). ``material`` gains
    drop_per_clear * clears_per_second, clears_per_second =
    1/clear_time, capped by stamina regen / stamina_per_clear when both are
    set (steady state: the stamina stock buffer is not modelled).
    """

    type: Literal["farm"] = "farm"
    name: str = ""
    material: str
    power_resource: str
    stages: list[FarmStage]
    clear_time: float = Field(default=1.0, gt=0)
    stamina_per_clear: float | None = Field(default=None, gt=0)
    stamina_regen: float | None = Field(default=None, ge=0)

    def clears_per_second(self) -> float:
        cps = 1.0 / self.clear_time
        if self.stamina_per_clear and self.stamina_regen is not None:
            cps = min(cps, self.stamina_regen / self.stamina_per_clear)
        return cps

    def stage_for(self, power: float) -> int | None:
        best = None
        for i, s in enumerate(self.stages):
            if power >= s.required_power and (best is None or s.drop_per_clear >= self.stages[best].drop_per_clear):
                best = i
        return best


NodeUnion = Annotated[
    Resource
    | Generator
    | NestedGenerator
    | Upgrade
    | PrestigeLayer
    | SacrificeNode
    | Achievement
    | Manager
    | Converter
    | ProbabilityNode
    | EndCondition
    | UnlockGate
    | ChoiceGroup
    | Register
    | Gate
    | Queue
    | TickspeedNode
    | AutobuyerNode
    | DrainNode
    | BuffNode
    | SynergyNode
    | FarmNode,
    Field(discriminator="type"),
]
