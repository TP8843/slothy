import itertools
from math import floor, ceil

from slothy.targets.riscv.riscv import RegisterType
from slothy.targets.riscv.riscv_instruction_core import RISCVInstruction

# TODO: Expand inputs and outputs for LMUL
# TODO: Add v0 as an input if the mask is selected
# TODO: Model vtype as input to vector instructions to stop invalid reordering

# TODO: When expanding registers for loading, need to actually use EMUL, not LMUL (EMUL = (EEW / SEW) * LMUL)

# LMUL Helper Methods
def _get_lmul_value(obj=None):
    """Get LMUL value from instruction object or any loaded RISC-V target module"""
    import sys

    # Try to get from instruction object first
    if obj is not None:
        lmul = getattr(obj, "lmul", None)
        if lmul is not None:
            return _parse_lmul_string(lmul)

    # Try to get from any loaded RISC-V target module
    for module_name, module in sys.modules.items():
        if (
            module_name.startswith("slothy.targets.riscv.")
            and hasattr(module, "lmul")
            and module.lmul is not None
        ):
            return _parse_lmul_string(module.lmul)

    return 1  # Default


def _parse_lmul_string(lmul):
    """Parse LMUL string (e.g., 'm2', 'm4', 'm8', 'mf2', 'mf4', 'mf8') to integer"""
    if isinstance(lmul, str):
        if lmul.startswith("m") and not lmul.startswith("mf"):
            lmul = int(lmul[1:])  # e.g., "m2" -> 2
        elif lmul.startswith("mf"):
            lmul = 1  # Fractional LMUL, treat as 1 for now
        else:
            lmul = 1

    # Ensure LMUL is valid
    if lmul not in [1, 2, 4, 8]:
        lmul = 1

    return lmul

def generate_expansion_factor(base_expansion_factor: int, local_expansion_factor: float) -> int:
    """Generate the final expansion factor for a vector register"""
    if local_expansion_factor == 0:
        return base_expansion_factor

    return max(1, ceil(base_expansion_factor * local_expansion_factor))


def _expand_vector_registers_generic(
    obj: any,
    base_expansion_factor: int,
    input_local_expansion_factors: list[float] = None,
    output_local_expansion_factors: list[float] = None,
    in_out_local_expansion_factors: list[float] = None,
) -> any:
    """
    Expand vector registers based on expansion factor for vector instructions.

    Groups consecutive vector registers together based on the expansion factor
    (LMUL or NF value):

      - With expansion=2: ``v8`` becomes [``v8, v9``], ``v4`` becomes [``v4, v5``]
      - With expansion=4: ``v8`` becomes [``v8, v9, v10, v11``]

    This function:

    #. Automatically detects which operands are vector registers
    #. Expands vector operands into register groups
    #. Preserves scalar/immediate operands unchanged
    #. Sets up constraint combinations for SLOTHY's register allocator
    #. Allows selective expansion of specific operands (useful for masked instructions)

    :param obj: Instruction object to modify
    :type obj: any
    :param base_expansion_factor: Base expansion value (usually LMUL or NF value)
    :type base_expansion_factor: int
    :param input_local_expansion_factors: Local expansion factors to multiply base factor by for inputs
    :type input_local_expansion_factors: list[float]
    :param output_local_expansion_factors: Local expansion factors to multiply base factor by for outputs
    :type input_local_expansion_factors: list[float]
    :param in_out_local_expansion_factors: Local expansion factors to multiply base factor by for in_outs
    :type input_local_expansion_factors: list[float]
    :return: modified obj
    :rtype: any
    """

    # Setup defaults
    if output_local_expansion_factors is None: output_local_expansion_factors = [1 for _ in range(len(obj.args_out))]
    if input_local_expansion_factors is None: input_local_expansion_factors = [1 for _ in range(len(obj.args_in))]
    if in_out_local_expansion_factors is None: in_out_local_expansion_factors = [1 for _ in range(len(obj.args_in_out))]

    if (base_expansion_factor <= 1 and
        all(f <= 1 for f in output_local_expansion_factors) and
        all(f <= 1 for f in input_local_expansion_factors) and
        all(f <= 1 for f in in_out_local_expansion_factors)):
        return obj

    available_regs = RegisterType.list_registers(RegisterType.VECT)

    def is_vector_register(reg):
        """Check if a register is a vector register."""
        return reg in available_regs

    def expand_vector_register(reg, expansion_factor):
        """Expand a vector register into a group of consecutive registers."""
        if not is_vector_register(reg):
            return [reg]  # Not a vector register, keep as-is

        start_idx = available_regs.index(reg)
        if start_idx + expansion_factor > len(available_regs):
            return [reg]  # Not enough consecutive registers, keep original

        return [available_regs[start_idx + i] for i in range(expansion_factor)]

    def expand_register_list(orig_args, orig_arg_types, local_expansion_factors):
        """Expand a list of registers, tracking expansion info for constraints.

        :param expand_indices: Indices to expand (None = expand all vectors)
        """
        expanded_args = []
        new_arg_types = []
        constraint_indices = []
        num_vectors = 0
        expanded_idx = 0

        for i, reg in enumerate(orig_args):
            should_expand = (
                is_vector_register(reg) and
                (
                        local_expansion_factors is None or # All should use base_expansion_factor
                        (i < len(local_expansion_factors) and local_expansion_factors[i] > 0) # Multiply by local_expansion_factor
                )
            )

            if should_expand:
                expanded_regs = expand_vector_register(reg, base_expansion_factor * local_expansion_factors[i])
                expanded_args.extend(expanded_regs)
                new_arg_types.extend([RegisterType.VECT] * len(expanded_regs))
                constraint_indices.extend(
                    range(expanded_idx, expanded_idx + len(expanded_regs))
                )
                expanded_idx += len(expanded_regs)
                num_vectors += 1
            else:
                expanded_args.append(reg)
                new_arg_types.append(orig_arg_types[i])
                expanded_idx += 1

        return expanded_args, new_arg_types, constraint_indices, num_vectors

    def generate_combinations(local_expansion_factor: float):
        """Generate all possible register group combinations (aligned groups)."""
        # TODO: Needs different constraints for each register depending on the local_expansion_factor
        final_expansion_factor = generate_expansion_factor(base_expansion_factor, local_expansion_factor)

        return [
            [available_regs[i + j] for j in range(final_expansion_factor)]
            for i in range(0, len(available_regs), final_expansion_factor)
            if i + base_expansion_factor <= len(available_regs)
        ]

    # Expand outputs, inputs, and in_outs
    expanded_outputs, new_arg_types_out, output_constraint_indices, _ = (
        expand_register_list(obj.args_out, obj.arg_types_out, output_local_expansion_factors)
    )
    expanded_inputs, new_arg_types_in, input_constraint_indices, num_vector_inputs = (
        expand_register_list(obj.args_in, obj.arg_types_in, input_local_expansion_factors)
    )
    expanded_in_outs, new_arg_types_in_out, in_out_constraint_indices, num_vector_in_outs = (
        expand_register_list(obj.args_in_out, obj.arg_types_in_out, in_out_local_expansion_factors)
    )

    if output_constraint_indices:
        obj.args_out_combinations = [(output_constraint_indices, generate_combinations(output_local_expansion_factors[0]))]

    if input_constraint_indices:
        # Generate combinations for multiple vector inputs using Cartesian product
        # TODO: Generate combinations for different sizes of input

        valid_combinations = [generate_combinations(input_local_expansion_factors[i]) for i, reg in enumerate(obj.args_in)]

        multi_combinations = [
            [reg for combo in combination for reg in combo]
            for combination in itertools.product(valid_combinations)
        ]
        obj.args_in_combinations = [(input_constraint_indices, multi_combinations)]

    if in_out_constraint_indices:
        valid_combinations = [generate_combinations(in_out_local_expansion_factors[i]) for i, reg in enumerate(obj.args_in_out)]

        multi_combinations = [
            [reg for combo in combination for reg in combo]
            for combination in itertools.product(valid_combinations)
        ]
        obj.in_out_combinations = [(in_out_constraint_indices, multi_combinations)]

    # Update instruction object
    obj.args_out = expanded_outputs
    obj.args_in = expanded_inputs
    obj.args_in_out = expanded_in_outs
    obj.num_out = len(expanded_outputs)
    obj.num_in = len(expanded_inputs)
    obj.num_in_out = len(expanded_in_outs)
    obj.arg_types_out = new_arg_types_out
    obj.arg_types_in = new_arg_types_in
    obj.arg_types_in_out = new_arg_types_in_out

    # Set up empty restrictions
    obj.args_out_restrictions = [None] * obj.num_out
    obj.args_in_restrictions = [None] * obj.num_in

    return obj

# TODO: Local expansion factors needs to be renamed to match the new registers
def _extract_base_registers(
    args_list: list, base_expansion_factor: int, local_expansion_factors: list[float]
) -> list:
    """Extract base registers from expanded register groups.

    :param args_list: List of register arguments
    :type args_list: list
    :param base_expansion_factor: LMUL or NF expansion factor
    :type base_expansion_factor: int
    :param local_expansion_factors: Factors for specific registers
    :type local_expansion_factors: list[float]
    :returns: List of base registers for display
    :rtype: list
    """
    if not args_list:
        return args_list.copy()

    display_args = []
    idx = 0
    local_expansion_factor_index = 0

    # Extract first register from each expandable group
    while idx < len(args_list):
        display_args.append(args_list[idx])

        # TODO: Check that this correctly extracts registers
        idx += generate_expansion_factor(base_expansion_factor, local_expansion_factors[local_expansion_factor_index])
        local_expansion_factor_index += 1

    return display_args


def _write_expanded_instruction(
    self: any,
    expansion_factor: int, # TODO: Handle local expansion factors (maybe)
    input_local_expansion_factors: list[float],
    output_local_expansion_factors: list[float],
    in_out_local_expansion_factors: list[float],
) -> any:
    """Custom write method for expanded instructions that shows only base registers.

    Works for both LMUL and NF expansion, handles cases with:

    - Only expanded outputs (load instructions)
    - Only expanded inputs (store instructions)
    - Both expanded inputs and outputs

    :param self: self
    :type self: any
    :param expansion_factor: The LMUL or NF expansion factor
    :type expansion_factor: int
    :param num_expandable_vector_inputs:
      Number of vector inputs that get expanded
      (excludes mask registers and other non-expandable vectors)
    :type num_expandable_vector_inputs: int
    :returns: Formatted instruction string with base registers only
    :rtype: any
    """
    # Early return for simple case
    if expansion_factor <= 1:
        return RISCVInstruction.write(self)

    # Check if we have expansion (either inputs or outputs)
    has_expansion = expansion_factor > 1
    has_expanded_inputs = (
        has_expansion and
        any(factor > 0 for factor in input_local_expansion_factors)
    )
    has_expanded_in_outs = (
        has_expansion and
        any(factor > 0 for factor in in_out_local_expansion_factors)
    )
    has_expanded_outputs = has_expansion and len(self.args_out) > 1

    if has_expanded_inputs or has_expanded_outputs or has_expanded_in_outs:
        out = self.pattern

        # Extract base registers for display
        display_args_out = _extract_base_registers(
            self.args_out,
            expansion_factor,
            output_local_expansion_factors,
        )
        display_args_in = _extract_base_registers(
            self.args_in,
            expansion_factor,
            input_local_expansion_factors
        )
        display_args_in_out = _extract_base_registers(
            self.args_in_out,
            expansion_factor,
            in_out_local_expansion_factors
        )

        l = (
            list(zip(display_args_in, self.pattern_inputs))
            + list(zip(display_args_out, self.pattern_outputs))
            + list(zip(display_args_in_out, self.pattern_in_outs))
        )

        for arg, (s, ty) in l:
            out = RISCVInstruction._instantiate_pattern(s, ty, arg, out)

        # Handle other pattern replacements
        def replace_pattern(txt, attr_name, mnemonic_key, t=None):
            def t_default(x):
                return x

            if t is None:
                t = t_default
            a = getattr(self, attr_name)
            if a is None and attr_name == "is32bit":
                return txt.replace("<w>", "")
            if a is None:
                return txt
            if not isinstance(a, list):
                txt = txt.replace(f"<{mnemonic_key}>", t(a))
                return txt
            for i, v in enumerate(a):
                txt = txt.replace(f"<{mnemonic_key}{i}>", t(v))
            return txt

        out = replace_pattern(out, "immediate", "imm", lambda x: f"{x}")
        out = replace_pattern(out, "datatype", "dt", lambda x: x.upper())
        out = replace_pattern(out, "flag", "flag")
        out = replace_pattern(out, "index", "index", str)
        out = replace_pattern(out, "is32bit", "w", lambda x: x.lower())
        out = replace_pattern(out, "len", "len")
        out = replace_pattern(out, "vm", "vm")
        out = replace_pattern(out, "vtype", "vtype")
        out = replace_pattern(out, "sew", "sew")
        out = replace_pattern(out, "lmul", "lmul")
        out = replace_pattern(out, "tpol", "tpol")
        out = replace_pattern(out, "mpol", "mpol")
        out = replace_pattern(out, "nf", "nf")
        out = replace_pattern(out, "ew", "ew")

        out = out.replace("\\[", "[")
        out = out.replace("\\]", "]")
        return out

    # Should not reach here, but fallback to default behavior
    return RISCVInstruction.write(self)



class v_set_vl_i(RISCVInstruction):
    pattern = "vsetvli <Xd>, <Xa>, <vtype>"
    inputs = ["Xa"]
    outputs = ["Xd"] # TODO: Model vtype in output

class v_i_set_vl_i(RISCVInstruction):
    pattern = "vsetivli <Xd>, <imm>, <vtype>"
    outputs = ["Xd"] # TODO: Model vtype in output

class v_set_vl(RISCVInstruction):
    pattern = "vsetvl <Xd>, <Xa>, <Xb>"
    inputs = ["Xa", "Xb"]
    outputs = ["Xd"] # TODO: Model vtype in output


class RISCVVectorInstruction(RISCVInstruction):
    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.make(src)
        lmul = _get_lmul_value(obj)
        return _expand_vector_registers_generic(obj, lmul)



class RISCVVectorVectorVectorVector(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <Va><vm>"
    inputs = ["Va", "Vb"]
    outputs = ["Vd"]

class RISCVVectorIntVectorVectorVector(RISCVVectorVectorVectorVector):
    pass

class RISCVVectorIntVectorVectorVectorPassthrough(RISCVVectorVectorVectorVector):
    outputs = []
    in_outs = ["Vd"]

class RISCVVectorIntVectorVectorVectorNarrowing(RISCVVectorIntVectorVectorVector):
    # TODO: Handle vs2 as 2*SEW
    pass

class RISCVVectorIntVectorVectorVectorWidening(RISCVVectorIntVectorVectorVector):
    # TODO: Handle the EMUL being twice the LMUL
    pass

class RISCVVectorIntVectorVectorVectorWideningPassthrough(RISCVVectorIntVectorVectorVector):
    # TODO: Handle Vd being 2*SEW in input and output
    outputs = []
    in_outs = ["Vd"]

class RISCVVectorIntVectorVectorVectorWideningVs2(RISCVVectorIntVectorVectorVector):
    # TODO: Handle the EMUL being twice the LMUL, vs2 = 2*SEW
    pass

class RISCVVectorFixedVectorVectorVector(RISCVVectorVectorVectorVector):
    pass

class RISCVVectorFixedVectorVectorVectorNarrowing(RISCVVectorFixedVectorVectorVector):
    # TODO: Handle vs2 being 2*SEW
    pass

class RISCVVectorMaskVectorVectorVector(RISCVVectorVectorVectorVector):
    pass



class RISCVVectorVectorVectorScalar(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <Xa><vm>"
    inputs = ["Vb", "Xa"]
    outputs = ["Vd"]

class RISCVVectorIntVectorVectorScalar(RISCVVectorVectorVectorScalar):
    pass

class RISCVVectorFixedVectorVectorScalar(RISCVVectorVectorVectorScalar):
    pass

class RISCVVectorFixedVectorVectorScalarNarrowing(RISCVVectorFixedVectorVectorScalar):
    # TODO: Handle vs2 being 2*SEW
    pass

class RISCVVectorIntVectorVectorScalarPassthrough(RISCVVectorVectorVectorScalar):
    outputs = []
    in_outs = ["Vd"]

class RISCVVectorIntVectorVectorScalarNarrowing(RISCVVectorIntVectorVectorScalar):
    # TODO: Handle the EMUL being half the LMUL
    pass

class RISCVVectorIntVectorVectorScalarWidening(RISCVVectorIntVectorVectorScalar):
    # TODO: Handle the EMUL being twice the LMUL
    pass

class RISCVVectorIntVectorVectorScalarWideningVs2(RISCVVectorIntVectorVectorScalar):
    # TODO: Handle the EMUL being twice the LMUL, vs2 = 2*SEW
    pass

class RISCVVectorIntVectorVectorScalarWideningPassthrough(RISCVVectorIntVectorVectorScalar):
    # TODO: Handle Vd being 2*SEW in input and output
    outputs = []
    in_outs = ["Vd"]



class RISCVVectorVectorVectorImmediate(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <Va>, <imm><vm>"
    inputs = ["Va"]
    outputs = ["Vd"]

class RISCVVectorIntVectorVectorImmediate(RISCVVectorVectorVectorImmediate):
    pass

class RISCVVectorIntVectorVectorImmediateNarrowing(RISCVVectorIntVectorVectorImmediate):
    # TODO: Handle vs2 being widened (vs2 = 2*SEW)
    pass

class RISCVVectorFixedVectorVectorImmediate(RISCVVectorVectorVectorImmediate):
    pass

class RISCVVectorFixedVectorVectorImmediateNarrowing(RISCVVectorFixedVectorVectorImmediate):
    # TODO: Handle vs2 being widened (vs2 = 2*SEW)
    pass



class RISCVVectorIntVectorVectorMask(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <Va>, <Ve>" # Ve is mask (so is v0)
    inputs = ["Va", "Vb", "Ve"]
    outputs = ["Vd"]

class RISCVVectorIntVectorScalarMask(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <Xa>, <Ve>" # Ve is mask (so is v0)
    inputs = ["Va", "Vb", "Ve"]

class RISCVVectorIntVectorImmediateMask(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <imm>, <Ve>" # Ve is mask (so is v0)
    inputs = ["Vb", "Ve"]

class RISCVVectorIntVectorMask(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <Vb><vm>"
    inputs = ["Vb"]
    output = ["Vd"]



class RISCVVectorMaskScalarVector(RISCVVectorInstruction):
    pattern = "mnemonic <Xd>, <Vb><vm>"
    inputs = ["Vb"]
    outputs = ["Xd"]

class RISCVVectorMaskVectorVector(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <Vb><vm>"
    inputs = ["Vb"]
    outputs = ["Vd"]

class RISCVVectorMaskVector(RISCVVectorInstruction):
    pattern = "mnemonic <Vd><vm>"
    in_outs = ["Vd"]

class RISCVVectorLoad(RISCVVectorInstruction):
    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.make(src)
        obj.increment = None
        obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]
        return obj

# TODO: Since len is referenced, calculate the EMUL for these instructions
class RISCVVectorUnitStrideLoad(RISCVVectorLoad):
    pattern = "mnemonic <Vd>, (<Xa>)<vm>"
    inputs = ["Xa"]
    outputs = ["Vd"]

class RISCVVectorUnitStrideMaskLoad(RISCVVectorLoad):
    pattern = "mnemonic <Vd>, (<Xa>)"
    inputs = ["Xa"]
    outputs = ["Vd"]

class RISCVVectorStrideLoad(RISCVVectorLoad):
    pattern = "mnemonic <Vd>, (<Xa>), <Xb><vm>"
    inputs = ["Xa", "Xb"]
    outputs = ["Vd"]

class RISCVVectorIndexedLoad(RISCVVectorLoad):
    # TODO: EMUL used for index register <Vb>, LMUL used for <Vd>
    pattern = "mnemonic <Vd>, (<Xa>), <Vb><vm>"
    inputs = ["Xa", "Vb"]
    outputs = ["Vd"]

class RISCVVectorSegmentLoad(RISCVVectorLoad):
    # TODO: Account for len defining the number of registers being used
    pattern = "mnemonic <Vd>, (<Xa>)<vm>"
    inputs = ["Xa"]
    outputs = ["Vd"]

class RISCVVectorStrideSegmentLoad(RISCVVectorLoad):
    # TODO: Account for len defining the number of registers being used
    pattern = "mnemonic <Vd>, (<Xa>), <Xb><vm>"
    inputs = ["Va", "Xa", "Xb"]

class RISCVVectorIndexedSegmentLoad(RISCVVectorLoad):
    # TODO: Account for len defining the number of registers being used
    pattern = "mnemonic <Vd>, (<Xa>), <Xb><vm>"
    inputs = ["Xa", "Xb"]
    outputs = ["Vd"]

class RISCVWholeVectorLoad(RISCVVectorLoad):
    pattern = "mnemonic <Vd>, (<Xa>)"
    inputs = ["Xa"]
    outputs = ["Vd"]

class RISCVVectorStore(RISCVVectorInstruction):
    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.make(src)
        obj.increment = None
        obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]
        return obj

class RISCVVectorUnitStrideStore(RISCVVectorStore):
    pattern = "mnemonic <Va>, (<Xa>)vm"
    inputs = ["Va", "Xa"]
    outputs = []

class RISCVVectorUnitStrideMaskStore(RISCVVectorStore):
    pattern = "mnemonic <Va>, (<Xa>)"
    inputs = ["Va", "Xa"]
    outputs = []

class RISCVVectorStrideStore(RISCVVectorStore):
    pattern = "mnemonic <Va>, (<Xa>), <Xb><vm>"
    inputs = ["Va", "Xa", "Xb"]

class RISCVVectorIndexedStore(RISCVVectorStore):
    # TODO: EMUL used for index register <Vb>, LMUL used for <Vd>
    pattern = "mnemonic <Va>, (<Xa>), <Xb><vm>"
    inputs = ["Va", "Xa", "Xb"]

class RISCVVectorSegmentStore(RISCVVectorStore):
    # TODO: Account for len defining the number of registers being used
    pattern = "mnemonic <Vd>, (<Xa>)<vm>"
    inputs = ["Xa"]
    outputs = ["Vd"]

class RISCVVectorStrideSegmentStore(RISCVVectorStore):
    # TODO: Account for len defining the number of registers being used
    pattern = "mnemonic <Va>, (<Xa>), <Xb><vm>"
    inputs = ["Va", "Xa", "Xb"]

class RISCVVectorIndexedSegmentStore(RISCVVectorStore):
    pattern = "mnemonic <Va>, (<Xa>), <Vb><vm>"
    inputs = ["Va", "Xa", "Vb"]

class RISCVVectorWholeVectorStore(RISCVVectorStore):
    pattern = "mnemonic <Va>, (<Xa>)"
    inputs = ["Va", "Xa"]


v_instrs = [
    (["vsetvli"], v_set_vl_i),
    (["visetvli"], v_i_set_vl_i),
    (["vsetvl"], v_set_vl),

    # Vector Load
    (
        [
            "vle<len>.v",
            "vle<len>ff.v",
        ],
        RISCVVectorUnitStrideLoad
    ),
    (
        [
            "vlm.v"
        ],
        RISCVVectorUnitStrideMaskLoad
    ),
    (
        [
            "vluxei<len>.v",
            "vloxei<len>.v"
         ],
        RISCVVectorIndexedLoad
    ),
    (
        [
            "vlse<len>.v",
        ],
        RISCVVectorStrideLoad
    ),
    (
        [
            "vlseg<nf>e<len>.v",
        ],
        RISCVVectorSegmentLoad,
    ),
    (
        [
            "vlsseg<nf>e<len>.v"
        ],
        RISCVVectorStrideSegmentLoad,
    ),
    (
        [
            "vluxseg<nf>ei<len>.v",
            "vloxseg<nf>ei<len>.v"
        ],
        RISCVVectorIndexedSegmentLoad
    ),
    (
        [
            "vl<nf>re<len>.v",
            "vl<nf>r.v"
        ],
        RISCVWholeVectorLoad
    ),

    # Vector Store
    (
        [
            "vse<len>.v",
        ],
        RISCVVectorUnitStrideStore
    ),
    (
        [
            "vsm.v"
        ],
        RISCVVectorUnitStrideMaskStore
    ),
    (
        [
            "vsse<len>.v",
        ],
        RISCVVectorStrideStore
    ),
    (
        [
            "vsuxei<len>.v",
            "vsoxei<len>.v"
        ],
        RISCVVectorIndexedStore
    ),
    (
        [
            "vsseg<nf>e<len>.v"
        ],
        RISCVVectorSegmentStore,
    ),
    (
        [
            "vssseg<nf>e<len>.v"
        ],
        RISCVVectorStrideSegmentStore,
    ),
    (
        [
            "vsuxseg<nf>ei<len>.v",
            "vsoxseg<nf>ei<len>.v"
        ],
        RISCVVectorIndexedSegmentStore
    ),
    (
        [
         "vs<nf>r.v"
        ],
        RISCVWholeVectorLoad
    ),

    (
        [

            # Vector Integer
            "vadd.vv",
            "vsub.vv",
            "vminu.vv",
            "vmin.vv",
            "vmaxu.vv",
            "vmax.vv",
            "vand.vv",
            "vor.vv",
            "vxor.vv",
            "vmsbc.vv",
            "vmseq.vv",
            "vmsne.vv",
            "vmsltu.vv",
            "vmsleu.vv",
            "vmsle.vv",
            "vdivu.vv",
            "vdiv.vv",
            "vremu.vv",
            "vrem.vv",
            "vmulhu.vv",
            "vsll.vv",
            "vmul.vv",
            "vmulh.vv"
            "vsrl.vv",
            "vsra.vv",
        ],
        RISCVVectorIntVectorVectorVector,
    ),
    (
        [
            "vmadd.vv",  # TODO: Ensure that Vd is also classed as an input
            "vnmsub.vv",  # TODO: Ensure that Vd is also classed as an input
            "vmacc.vv",  # TODO: Ensure that Vd is also classed as an input
            "vnmsac.vv",  # TODO: Ensure that Vd is also classed as an input
        ],
        RISCVVectorIntVectorVectorVectorPassthrough
    ),
    (
        [
            "vwaddu.vv",  # TODO: Ensure widening is tracked
            "vwadd.vv",  # TODO: Ensure widening is tracked
            "vwsubu.vv",  # TODO: Ensure widening is tracked
            "vwsub.vv",  # TODO: Ensure widening is tracked
            "vwmulu.vv",  # TODO: Ensure widening is tracked
            "vwmulsu.vv",  # TODO: Ensure widening is tracked
            "vwmul.vv",  # TODO: Ensure widening is tracked
        ],
        RISCVVectorIntVectorVectorVectorWidening
    ),
    (
        [
            "vwaddu.wv",  # TODO: Ensure widening is tracked, vs2 is also 2*SEW
            "vwadd.wv",  # TODO: Ensure widening is tracked, vs2 is also 2*SEW
            "vwsubu.wv",  # TODO: Ensure widening is tracked, vs2 is also 2*SEW
            "vwsub.wv",  # TODO: Ensure widening is tracked, vs2 is also 2*SEW
        ],
        RISCVVectorIntVectorVectorVectorWideningVs2,
    ),
    (
        [
            "vwmaccu.vv", # TODO: Ensure widening is tracked, TODO: Ensure that Vd is also classed as an input, vd is also 2*SEW
            "vwmacc.vv", # TODO: Ensure widening is tracked, TODO: Ensure that Vd is also classed as an input, vd is also 2*SEW
            "vwmaccsu.vv", # TODO: Ensure widening is tracked, TODO: Ensure that Vd is also classed as an input, vd is also 2*SEW
        ],
        RISCVVectorIntVectorVectorVectorWideningPassthrough,
    ),
    (
        [
            "vnsrl.wv",  # TODO: Ensure narrowing is tracked, vs2 is 2*SEW
            "vnsra.wv",  # TODO: Ensure narrowing is tracked, vs2 is 2*SEW
        ],
        RISCVVectorIntVectorVectorVectorNarrowing
    ),


    (
        [
            "vaaddu.vv",
            "vaadd.vv",
            "vasubu.vv",
            "vasub.vv",
            "vsaddu.vv",
            "vsadd.vv",
            "vssubu.vv",
            "vssub.vv",
            "vsmul.vv",
            "vssrl.vv",
            "vssra.vv",
        ],
        RISCVVectorFixedVectorVectorVector,
    ),
    (
        [
            "vnclipu.wv",  # TODO: Only vs2 is widened
            "vnclip.wv",  # TODO: Only vs2 is widened
        ],
        RISCVVectorFixedVectorVectorVectorNarrowing
    ),
    (
        [
            # Vector Integer
            "vadd.vx",
            "vsub.vx",
            "vrsub.vx",
            "vminu.vx",
            "vmin.vx",
            "vmaxu.vx",
            "vmax.vx",
            "vand.vx",
            "vor.vx",
            "vxor.vx",
            "vmsbc.vx",
            "vmseq.vx",
            "vmsne.vx",
            "vmsltu.vx",
            "vmsleu.vx",
            "vmsle.vx",
            "vmsgtu.vx",
            "vmsgt.vx",
            "vdivu.vx",
            "vdiv.vx",
            "vremu.vx",
            "vmulhu.vx",
            "vsll.vx",
            "vmul.vx",
            "vmulh.vx",
            "vsrl.vx",
            "vsra.vx",
        ],
        RISCVVectorIntVectorVectorScalar,
    ),
    (
        [
            "vmadd.vx",  # TODO: Ensure that Vd is also classed as an input
            "vnmsub.vx",  # TODO: Ensure that Vd is also classed as an input
            "vmacc.vx",  # TODO: Ensure that Vd is also classed as an input
            "vnmsac.vx",  # TODO: Ensure that Vd is also classed as an input
        ],
        RISCVVectorIntVectorVectorScalarPassthrough,
    ),
    (
        [
            "vnsrl.wx",  # TODO: Ensure narrowing is tracked (vs2 = 2*SEW)
            "vnsra.wx",  # TODO: Ensure narrowing is tracked (vs2 = 2*SEW)
        ],
        RISCVVectorIntVectorVectorScalarNarrowing
    ),
    (
        [
            "vwaddu.vx",  # TODO: Ensure widening is tracked
            "vwadd.vx",  # TODO: Ensure widening is tracked
            "vwsubu.vx",  # TODO: Ensure widening is tracked
            "vwsub.vx",  # TODO: Ensure widening is tracked
            "vwmulu.vx",  # TODO: Ensure widening is tracked
            "vwmulsu.vx",  # TODO: Ensure widening is tracked
            "vwmul.vx",  # TODO: Ensure widening is tracked
        ],
        RISCVVectorIntVectorVectorScalarWidening,
    ),
    (
        [
            "vwaddu.wx",  # TODO: Ensure widening is tracked, vs2 is also 2*SEW
            "vwadd.wx",  # TODO: Ensure widening is tracked, vs2 is also 2*SEW
            "vwsubu.wx",  # TODO: Ensure widening is tracked, vs2 is also 2*SEW
            "vwsub.wx",  # TODO: Ensure widening is tracked, vs2 is also 2*SEW
        ],
        RISCVVectorIntVectorVectorScalarWideningVs2,
    ),
    (
        [
            "vwmaccu.vx", # TODO: Ensure widening is tracked, TODO: Ensure that Vd is also classed as an input (vd = 2*SEW)
            "vwmacc.vx", # TODO: Ensure widening is tracked, TODO: Ensure that Vd is also classed as an input (vd = 2*SEW)
            "vwmaccus.vx", # TODO: Ensure widening is tracked, TODO: Ensure that Vd is also classed as an input (vd = 2*SEW)
            "vwmaccsu.vx", # TODO: Ensure widening is tracked, TODO: Ensure that Vd is also classed as an input (vd = 2*SEW)
        ],
        RISCVVectorIntVectorVectorScalarWideningPassthrough,
    ),


    (
        [
            "vaaddu.vx",
            "vaadd.vx",
            "vasubu.vx",
            "vasub.vx",
            "vsaddu.vx",
            "vsadd.vx",
            "vssubu.vx",
            "vssub.vx",
            "vsmul.vx",
            "vssrl.vx",
            "vssra.vx",
        ],
        RISCVVectorFixedVectorVectorScalar,
    ),
    (
        [
            "vnclipu.wx",  # TODO: Only vs2 is widened
            "vnclip.wx",  # TODO: Only vs2 is widened
        ],
        RISCVVectorFixedVectorVectorScalarNarrowing
    ),


    (
        [
            # Vector Integer
            "vadd.vi",
            "vrsub.vi",
            "vand.vi",
            "vor.vi",
            "vxor.vi",
            "vmseq.vi",
            "vmsne.vi",
            "vmsleu.vi",
            "vmsle.vi",
            "vmsgtu.vi",
            "vmsgt.vi",
            "vsll.vi",
            "vsrl.vi",
            "vsra.vi",
        ],
        RISCVVectorIntVectorVectorImmediate,
    ),
    (
        [
            "vnsrl.wi",  # TODO: Only vs2 is widened
            "vnsra.wi",  # TODO: Only vs2 is widened
        ],
        RISCVVectorIntVectorVectorImmediateNarrowing
    ),
    (
        [
            "vsaddu.vi",
            "vsadd.vi",
            "vssrl.vi",
            "vssra.vi",
        ],
        RISCVVectorFixedVectorVectorImmediate,
    ),
    (
        [
            "vnclipu.wi",  # TODO: Only vs2 is widened
            "vnclip.wi",  # TODO: Only vs2 is widened
        ],
        RISCVVectorFixedVectorVectorImmediateNarrowing,
    ),


    #  Mask Operations
    (
        [
            "vmsbf.m",
            "vmsof.m",
            "vmsif.m",
            "viota.m",
        ],
        RISCVVectorMaskVectorVector,
    ),
    (
        [
            "vcpop.m",
            "vfirst.m",
        ],
        RISCVVectorMaskScalarVector,
    ),
    (
        [
            "vid.v",
        ],
        RISCVVectorMaskVector,
    ),
    (
        [
            "vmandnot.mm",
            "vmand.mm",
            "vmor.mm",
            "vmxor.mm",
            "vmornot.mm",
            "vmnand.mm",
            "vmnor.mm",
            "vmornot.mm",
            "vmnand.mm",
            "vmnor.mm",
            "vmxnor.mm",
        ],
        RISCVVectorMaskVectorVectorVector,
    ),
    (
        [
            "vadc.vvm",
            "vmadc.vvm",
            "vsbc.vvm",
            "vmsbc.vvm",
        ],
        RISCVVectorIntVectorVectorMask,
    ),
    (
        [
            "vadc.vxm",
            "vmadc.vxm",
            "vsbc.vxm",
            "vmsbc.vxm",
        ],
        RISCVVectorIntVectorScalarMask,
    ),
    (
        [
            "vadc.vim",
            "vmadc.vim",
        ],
        RISCVVectorIntVectorImmediateMask,
    ),
    (
        [
            "vzext.vf<nf>",
             "vsext.vf<nf>",
        ],
        RISCVVectorIntVectorMask,
    ),
]

def generate_rv32_64_v_instructions():
    """
    Generates all instruction classes for the rv32_64_v instruction set
    """
    for elem in v_instrs:
        RISCVInstruction.instr_factory(elem[0], elem[1])

    RISCVInstruction.classes_by_names.update(
        {cls.__name__: cls for cls in RISCVInstruction.dynamic_instr_classes}
    )
    return RISCVInstruction.dynamic_instr_classes

generate_rv32_64_v_instructions()