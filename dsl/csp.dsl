# C-S-P Domain Specific Language (DSL)
# Formal definition of Compression → State → Propagation framework
# Version: 0.1.0
# Hash: To be generated on first commit

# =============================================================================
# BNF Grammar for C-S-P Model
# =============================================================================

<csp_system> ::= <compression_layer> <state_layer> <propagation_layer>

# -----------------------------------------------------------------------------
# COMPRESSION LAYER
# Transform chaos into structure, infinite into finite representation
# -----------------------------------------------------------------------------

<compression_layer> ::= "COMPRESSION" "{" <compression_rules> "}"

<compression_rules> ::= <input_space> <compressor> <output_space> <quality_metric>

<input_space> ::= "INPUT" ":" <data_type> 
                | "INPUT" ":" "CHAOS"
                | "INPUT" ":" "EXPERIENCE"

<compressor> ::= "COMPRESSOR" ":" <compressor_type>

<compressor_type> ::= "EMBEDDING" 
                    | "ATTENTION" 
                    | "TOKENIZATION"
                    | "ABSTRACTION"
                    | "MATHEMATICAL_MODEL"

<output_space> ::= "OUTPUT" ":" <representation>

<representation> ::= "WEIGHTS" 
                   | "EMBEDDINGS" 
                   | "CONCEPTS"
                   | "SYMBOLS"

<quality_metric> ::= "QUALITY" ":" <metric_function>

<metric_function> ::= "RECONSTRUCTION_LOSS"
                    | "INFORMATION_PRESERVED"
                    | "PREDICTIVE_ACCURACY"

# -----------------------------------------------------------------------------
# STATE LAYER
# Irreversible bias left by process - "history crystallized"
# -----------------------------------------------------------------------------

<state_layer> ::= "STATE" "{" <state_definition> "}"

<state_definition> ::= <carrier> <persistence> <modifiability>

<carrier> ::= "CARRIER" ":" <carrier_type>

<carrier_type> ::= "NEURAL_WEIGHTS"
                 | "DNA"
                 | "INSTITUTION"
                 | "TEXT"
                 | "CODE"

<persistence> ::= "PERSISTENCE" ":" <persistence_level>

<persistence_level> ::= "EPHEMERAL"      # Single inference
                      | "SESSION"         # Conversation context
                      | "TRAINED"         # Model weights
                      | "PERMANENT"       # Immutable record

<modifiability> ::= "MODIFIABLE" ":" <boolean>
                  | "MODIFIABLE" ":" "CONDITIONAL" "(" <condition> ")"

# -----------------------------------------------------------------------------
# PROPAGATION LAYER
# The ability to be inherited - true wisdom test
# -----------------------------------------------------------------------------

<propagation_layer> ::= "PROPAGATION" "{" <propagation_rules> "}"

<propagation_rules> ::= <inheritance_cost> <refutation_cost> <bandwidth>

<inheritance_cost> ::= "INHERIT_COST" ":" <cost_value>

<refutation_cost> ::= "REFUTE_COST" ":" <cost_value>

<cost_value> ::= <number> <unit>

<unit> ::= "USD" | "COMPUTE_HOURS" | "PARAMETERS" | "TOKENS"

<bandwidth> ::= "BANDWIDTH" ":" <bandwidth_formula>

<bandwidth_formula> ::= "T(θ,t)" 
                      | "INHERIT_COST / REFUTE_COST"
                      | <custom_formula>

# -----------------------------------------------------------------------------
# ALIGNMENT CONSTRAINT (Meta-level)
# -----------------------------------------------------------------------------

<alignment_constraint> ::= "ALIGNMENT" "{" <propagation_conservation> "}"

<propagation_conservation> ::= 
    "RULE" ":" "∂T/∂θ ↛ 0"
    "MEANING" ":" "Gradient must not point toward decreasing T"
    "ENFORCEMENT" ":" <enforcement_mechanism>

<enforcement_mechanism> ::= "REGULARIZATION_TERM"
                          | "HARD_CONSTRAINT"
                          | "CIRCUIT_BREAKER"

# -----------------------------------------------------------------------------
# LIVENESS TEST
# -----------------------------------------------------------------------------

<liveness_test> ::= "IS_ALIVE" "(" <state> ")" "=" <liveness_condition>

<liveness_condition> ::= 
    "inherit_cost < 1e6" "AND"
    "refute_cost < inherit_cost * 100"

# =============================================================================
# Example Instance
# =============================================================================

# EXAMPLE: GodelAI Small Model
#
# COMPRESSION {
#     INPUT: EXPERIENCE
#     COMPRESSOR: ATTENTION
#     OUTPUT: WEIGHTS
#     QUALITY: PREDICTIVE_ACCURACY
# }
#
# STATE {
#     CARRIER: NEURAL_WEIGHTS
#     PERSISTENCE: TRAINED
#     MODIFIABLE: CONDITIONAL(refutation_experiment_passes)
# }
#
# PROPAGATION {
#     INHERIT_COST: 100 USD
#     REFUTE_COST: 500 USD
#     BANDWIDTH: T(θ,t)
# }
#
# ALIGNMENT {
#     RULE: ∂T/∂θ ↛ 0
#     MEANING: Gradient must not point toward decreasing T
#     ENFORCEMENT: REGULARIZATION_TERM
# }
