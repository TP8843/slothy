import itertools
from math import ceil

from slothy.helper import AsmAllocation, SourceLine
from slothy.targets.riscv.riscv import RegisterType
from slothy.targets.riscv.riscv_instruction_core import RISCVInstruction

# TODO: Add v0 as an input if the mask is selected
# TODO: Model vtype as input to vector instructions to stop invalid reordering
# TODO: Vector Integer Compare instructions always use unexpanded destination

# LMUL Helper Methods
def _get_lmul_value(obj=None):
    """Get LMUL value from instruction object"""

    # Try to get from instruction object first
    if obj is not None:
        lmul = getattr(obj, "lmul", None)
        if lmul is not None:
            return _parse_lmul_string(lmul)

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

# SEW Helper Methods
def _get_sew_value(obj=None):
    """Get SEW value from instruction object"""

    # Try to get from instruction object first
    if obj is not None:
        sew = getattr(obj, "sew", None)
        if sew is not None:
            return _parse_sew_string(sew)

    return 32  # Default


def _parse_sew_string(sew):
    """Parse SEW string (e.g., 'e8', 'e16', 'e32', 'e64', 'e128', 'e256', 'e512', 'e1024') to integer"""
    if isinstance(sew, str):
        if sew.startswith("e"):
            sew = int(sew[1:])  # e.g., "e8" -> 8
        else:
            sew = 1

    # Ensure SEW is valid
    if sew not in [8, 16, 32, 64, 128, 256, 512, 1024]:
        sew = 32

    return sew

def generate_expansion_factor(base_expansion_factor: int, local_expansion_factor: float) -> int:
    """Generate the final expansion factor for a vector register"""
    if local_expansion_factor == 0:
        return 1 # Do not do any expansion of the local expansion factor is 0

    return max(1, ceil(base_expansion_factor * local_expansion_factor))


def expand_vector_register(reg, expansion_factor, available_regs):
    expansion_factor = ceil(expansion_factor)

    """Expand a vector register into a group of consecutive registers."""
    if reg not in available_regs:
        return [reg]  # Not a vector register, keep as-is

    start_idx = available_regs.index(reg)
    if start_idx + expansion_factor > len(available_regs):
        return [reg]  # Not enough consecutive registers, keep original

    return [available_regs[start_idx + i] for i in range(expansion_factor)]


def expand_register_list(orig_args, orig_arg_types, expansion_factors, available_regs, orig_in_out_differences = None, register_set: str = "in_out"):
    """Expand a list of registers, tracking expansion info for constraints.

    :param expansion_factors: Factors to expand by for each register
    :param register_set: Which set of registers is being processed (input, output, in_out)
    """
    expanded_args = []
    new_arg_types = []
    new_in_out_differences = None
    constraint_indices = []
    expanded_idx = 0

    if orig_in_out_differences is not None:
        new_in_out_differences = orig_in_out_differences.copy() if register_set == "in_out" else []


    for i, reg in enumerate(orig_args):
        should_expand = orig_arg_types[i] == RegisterType.VECT and expansion_factors[i] > 1

        if should_expand:
            expanded_regs = expand_vector_register(reg, expansion_factors[i], available_regs)
            expanded_args.extend(expanded_regs)
            new_arg_types.extend([orig_arg_types[i]] * len(expanded_regs))

            if orig_in_out_differences is not None and register_set == "output":
                new_in_out_differences.extend(
                    [[(j, input_reg) for j in range(expanded_idx, expanded_idx + len(expanded_regs))] for
                     (output_reg, input_reg) in orig_in_out_differences if output_reg == i])
            elif orig_in_out_differences is not None and register_set == "input":
                new_in_out_differences.extend(
                    [[(output_reg, i) for i in range(expanded_idx, expanded_idx + len(expanded_regs))] for
                     (output_reg, input_reg) in orig_in_out_differences if input_reg == i])

            constraint_indices.extend(
                range(expanded_idx, expanded_idx + len(expanded_regs))
            )
            expanded_idx += len(expanded_regs)
        else:
            expanded_args.append(reg)
            new_arg_types.append(orig_arg_types[i])
            if orig_in_out_differences is not None and register_set == "output":
                new_in_out_differences.extend([value for value in orig_in_out_differences if value[0] == i])
            elif orig_in_out_differences is not None and register_set == "input":
                new_in_out_differences.extend([value for value in orig_in_out_differences if value[1] == i])
            expanded_idx += 1

    return expanded_args, new_arg_types, new_in_out_differences, constraint_indices

def generate_combinations(expansion_factor: int, available_regs: list[str]):
    """Generate all possible register group combinations (aligned groups)."""

    return [
        [available_regs[i + j] for j in range(expansion_factor)]
        for i in range(0, 32, expansion_factor)
        # if i + expansion_factor <= len(available_regs)
    ]

def generate_multi_combinations(expansion_factors: list[int], register_types: list[RegisterType], available_regs: list[str]):
    """Generate all possible register group combinations for multiple registers"""

    valid_combinations = [
        generate_combinations(
            factor,
            available_regs
        )
        for i, factor in enumerate(expansion_factors)
        if factor > 1 and register_types[i] == RegisterType.VECT
    ]

    return [
        [reg for combo in combination for reg in combo]
        for combination in itertools.product(*valid_combinations)
    ]

def _expand_vector_registers_generic(
    obj: any,
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
    :return: modified obj
    :rtype: any
    """

    output_local_expansion_factors = obj.output_local_expansion_factors
    input_local_expansion_factors = obj.input_local_expansion_factors
    in_out_local_expansion_factors = obj.in_out_local_expansion_factors
    base_expansion_factor = obj.lmul_external

    if base_expansion_factor is None:
        base_expansion_factor = 1

    if (base_expansion_factor <= 1 and
        len(output_local_expansion_factors) > 0 and all(f <= 1 for f in output_local_expansion_factors) and
        len(input_local_expansion_factors) > 0 and all(f <= 1 for f in input_local_expansion_factors) and
        len(in_out_local_expansion_factors) > 0 and all(f <= 1 for f in in_out_local_expansion_factors)):
        return obj

    final_input_expansion_factors = [ceil(base_expansion_factor * factor) for factor in input_local_expansion_factors]
    final_output_expansion_factors = [ceil(base_expansion_factor * factor) for factor in output_local_expansion_factors]
    final_in_out_expansion_factors = [ceil(base_expansion_factor * factor) for factor in in_out_local_expansion_factors]

    available_regs = RegisterType.list_registers(RegisterType.VECT)

    # Expand outputs, inputs, and in_outs
    expanded_outputs, new_arg_types_out, new_different_constraints, output_constraint_indices = (
        expand_register_list(
            obj.args_out,
            obj.arg_types_out,
            final_output_expansion_factors,
            available_regs,
            orig_in_out_differences=obj.args_in_out_different,
            register_set="output"
        )
    )
    expanded_inputs, new_arg_types_in, new_different_constraints, input_constraint_indices = (
        expand_register_list(
            obj.args_in,
            obj.arg_types_in,
            final_input_expansion_factors,
            available_regs,
            orig_in_out_differences=new_different_constraints,
            register_set="input"
        )
    )
    expanded_in_outs, new_arg_types_in_out, new_different_constraints, in_out_constraint_indices = (
        expand_register_list(
            obj.args_in_out,
            obj.arg_types_in_out,
            final_in_out_expansion_factors,
            available_regs,
            orig_in_out_differences=new_different_constraints,
            register_set="in_out"
        )
    )

    if output_constraint_indices:
        obj.args_out_combinations = [
            (
                output_constraint_indices,
                generate_combinations(
                    final_output_expansion_factors[0],
                    available_regs
                )
            )
        ]

    if input_constraint_indices:
        # Generate combinations for multiple vector inputs using Cartesian product
        multi_combinations = generate_multi_combinations(
            final_input_expansion_factors,
            obj.arg_types_in,
            available_regs
        )

        obj.args_in_combinations = [(input_constraint_indices, multi_combinations)]

        if any(len(combination) != len(input_constraint_indices) for combination in multi_combinations):
            print("Does not match :( for input")
            print(f"instruction: {obj}")
            print(f"arg_types_in: {new_arg_types_in}")
            print(f"final_input expansion_factors: {final_input_expansion_factors}")
            print(f"new_different_constraints: {new_different_constraints}")
            print(f"original_inputs: {obj.args_in}")
            print(f"expanded_inputs: {expanded_inputs}")
            print(f"input_constraint_indices: {input_constraint_indices}")
            print(f"multi_combinations: {multi_combinations}")

    if in_out_constraint_indices:
        multi_combinations = generate_multi_combinations(
            final_in_out_expansion_factors,
            obj.arg_types_in_out,
            available_regs
        )
        obj.in_out_combinations = [(in_out_constraint_indices, multi_combinations)]

        if any(len(combination) != len(in_out_constraint_indices) for combination in multi_combinations):
            print("Does not match :( for in out")
            print(f"in_out_constraint_indices: {in_out_constraint_indices}")
            print(f"multi_combinations: {multi_combinations}")

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
    obj.args_in_out_different = new_different_constraints

    # Set up empty restrictions
    obj.args_out_restrictions = [None] * obj.num_out
    obj.args_in_restrictions = [None] * obj.num_in
    obj.args_in_out_restrictions = [None] * obj.num_in_out

    return obj

# TODO: Check that the register ordering stays consistent after optimisation
def _extract_base_registers(
    args_list: list, arg_types: list, expansion_factors: list[float]
) -> list:
    """Extract base registers from expanded register groups.

    :param args_list: List of register arguments
    :type args_list: list
    :param arg_types: Types for each register
    :type arg_types: RegisterType
    :param expansion_factors: Factors for specific registers
    :type expansion_factors: list[float]
    :returns: List of base registers for display
    :rtype: list
    """
    if not args_list:
        return args_list.copy()

    display_args = []
    idx = 0
    expansion_factor_index = 0

    # Extract first register from each expandable group
    while idx < len(args_list):
        display_args.append(args_list[idx])

        if (expansion_factors[expansion_factor_index] == 0 or
            arg_types[idx] != RegisterType.VECT):
            expansion_factor_index += 1
            idx += 1
        else:
            idx += expansion_factors[expansion_factor_index]
            expansion_factor_index += 1

    return display_args


def _write_expanded_instruction(
    self: any
) -> any:
    """Custom write method for expanded instructions that shows only base registers.

    Works for both LMUL and NF expansion, handles cases with:

    - Only expanded outputs (load instructions)
    - Only expanded inputs (store instructions)
    - Both expanded inputs and outputs

    :param self: self
    :type self: any
    :returns: Formatted instruction string with base registers only
    :rtype: any
    """

    output_local_expansion_factors = self.output_local_expansion_factors
    input_local_expansion_factors = self.input_local_expansion_factors
    in_out_local_expansion_factors = self.in_out_local_expansion_factors
    expansion_factor = self.lmul_external

    final_output_expansion_factors = [
        generate_expansion_factor(expansion_factor, factor)
        for factor in output_local_expansion_factors
    ]
    final_input_expansion_factors = [
        generate_expansion_factor(expansion_factor, factor)
        for factor in input_local_expansion_factors
    ]
    final_in_out_expansion_factors = [
        generate_expansion_factor(expansion_factor, factor)
        for factor in in_out_local_expansion_factors
    ]

    if expansion_factor is None:
        expansion_factor = 1

    # Early return for simple case
    if (
            all(factor <= 1 for i, factor in enumerate(final_input_expansion_factors)) and
            all(factor <= 1 for factor in final_output_expansion_factors) and
            all(factor <= 1 for factor in final_in_out_expansion_factors)
    ):
        return RISCVInstruction.write(self)

    # Check if we have expansion (either inputs or outputs)
    has_expansion = expansion_factor > 1
    has_expanded_inputs = any(factor > 0 for factor in final_input_expansion_factors)
    has_expanded_in_outs = any(factor > 0 for factor in in_out_local_expansion_factors)
    has_expanded_outputs = any(factor > 0 for factor in output_local_expansion_factors)

    if has_expanded_inputs or has_expanded_outputs or has_expanded_in_outs:
        out = self.pattern

        # Extract base registers for display
        display_args_out = _extract_base_registers(
            self.args_out,
            self.arg_types_out,
            final_output_expansion_factors
        )
        display_args_in = _extract_base_registers(
            self.args_in,
            self.arg_types_in,
            final_input_expansion_factors
        )
        display_args_in_out = _extract_base_registers(
            self.args_in_out,
            self.arg_types_in_out,
            final_in_out_expansion_factors
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


class RISCVVectorInstruction(RISCVInstruction):
    lmul_global = 1
    sew_global = 32


    def write(self):
        return _write_expanded_instruction(self)

    @classmethod
    def build(cls, c, src, expand_registers=True):
        obj = RISCVInstruction.build(c, src)
        # Store local config for each instruction
        obj.lmul_external = RISCVVectorInstruction.lmul_global
        obj.sew_external = RISCVVectorInstruction.sew_global

        if expand_registers:
            obj = _expand_vector_registers_generic(obj)

        obj.args_in.append("vtype")
        obj.arg_types_in.append(RegisterType.CSR)
        obj.args_in_restrictions.append(None)
        obj.num_in += 1
        if obj.input_local_expansion_factors is not None:
            obj.input_local_expansion_factors.append(0)

        return obj

    @classmethod
    def make(cls, src):
        return RISCVVectorInstruction.build(cls, src)

class RISCVVectorSetVtype(RISCVVectorInstruction):
    @classmethod
    def prepare_vtype(cls, slothy, start):
        """Prepare the vtype for RISC-V Vector instructions"""
        from slothy.helper import AsmHelper

        pre, body, post = AsmHelper.extract(slothy.source, start)
        body = AsmAllocation.unfold_all_aliases(slothy.config.register_aliases, body)
        body = SourceLine.split_semicolons(body)

        for line in body[0:5]:
            if line.text.strip() == "": continue
            instruction = RISCVInstruction.parser(line)
            # Initializing the instruction updates the vtype values
            if any(isinstance(inst, RISCVVectorSetVtype) for inst in instruction):
                return


    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)

        new_sew = getattr(obj, "sew", None)
        if new_sew is not None:
            RISCVVectorInstruction.sew_global = _parse_sew_string(new_sew)

        new_lmul = getattr(obj, "lmul", None)
        if new_lmul is not None:
            RISCVVectorInstruction.lmul_global = _parse_lmul_string(new_lmul)

        obj.lmul_external = RISCVVectorInstruction.lmul_global
        obj.sew_external = RISCVVectorInstruction.sew_global

        obj.args_out.append("vtype")
        obj.arg_types_out.append(RegisterType.CSR)
        obj.args_out_restrictions.append(None)
        obj.num_out += 1
        if obj.output_local_expansion_factors is not None:
            obj.output_local_expansion_factors.append(0)

        return obj

class vset_vl_i(RISCVVectorSetVtype):
    pattern = "mnemonic <Xd>, <Xa>, <vtype>"
    inputs = ["Xa"]
    outputs = ["Xd"]

class vset_i_vl_i(RISCVVectorSetVtype):
    pattern = "mnemonic <Xd>, <imm>, <vtype>"
    outputs = ["Xd"]

class vset_vl(RISCVVectorSetVtype):
    pattern = "mnemonic <Xd>, <Xa>, <Xb>"
    inputs = ["Xa", "Xb"]
    outputs = ["Xd"]


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
    input_local_expansion_factors = [1, 2]

class RISCVVectorIntVectorVectorVectorWidening(RISCVVectorIntVectorVectorVector):
    output_local_expansion_factors = [2]

class RISCVVectorIntVectorVectorVectorWideningPassthrough(RISCVVectorIntVectorVectorVector):
    # Handle Vd being 2*SEW in input and output
    outputs = []
    in_outs = ["Vd"]

    in_out_local_expansion_factors = [2]

class RISCVVectorIntVectorVectorVectorWideningVs2(RISCVVectorIntVectorVectorVector):
    input_local_expansion_factors = [1, 2]
    output_local_expansion_factors = [2]

class RISCVVectorFixedVectorVectorVector(RISCVVectorVectorVectorVector):
    pass

class RISCVVectorFixedVectorVectorVectorNarrowing(RISCVVectorFixedVectorVectorVector):
    input_local_expansion_factors = [1, 2]

class RISCVVectorMaskVectorVectorVector(RISCVVectorVectorVectorVector):
    pass

class RISCVVectorPermutationVectorVectorVector(RISCVVectorVectorVectorVector):
    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.args_in_out_different = [
            (0, 0),
            (0, 1),
        ]

        obj = _expand_vector_registers_generic(obj)
        return obj

class RISCVVectorPermutationVectorVectorVectorGatherE16(RISCVVectorPermutationVectorVectorVector):
    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.input_local_expansion_factors = [16.0 / obj.sew_external, 1]
        obj.args_in_out_different = [
            (0, 0),
            (0, 1),
        ]

        obj = _expand_vector_registers_generic(obj)
        return obj

class RISCVVectorVectorVectorScalar(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <Xa><vm>"
    inputs = ["Xa", "Vb"]
    outputs = ["Vd"]



class RISCVVectorIntVectorVectorScalar(RISCVVectorVectorVectorScalar):
    pass

class RISCVVectorIntVectorVectorScalarPassthrough(RISCVVectorVectorVectorScalar):
    outputs = []
    in_outs = ["Vd"]

class RISCVVectorIntVectorVectorScalarNarrowing(RISCVVectorIntVectorVectorScalar):
    input_local_expansion_factors = [1, 2]

class RISCVVectorIntVectorVectorScalarWidening(RISCVVectorIntVectorVectorScalar):
    output_local_expansion_factors = [2]

class RISCVVectorIntVectorVectorScalarWideningVs2(RISCVVectorIntVectorVectorScalar):
    input_local_expansion_factors = [1, 2]
    output_local_expansion_factors = [2]

class RISCVVectorIntVectorVectorScalarWideningPassthrough(RISCVVectorIntVectorVectorScalar):
    # Handle Vd being 2*SEW in input and output
    outputs = []
    in_outs = ["Vd"]

    in_out_local_expansion_factors = [2]



class RISCVVectorFixedVectorVectorScalar(RISCVVectorVectorVectorScalar):
    pass

class RISCVVectorFixedVectorVectorScalarNarrowing(RISCVVectorFixedVectorVectorScalar):
    input_local_expansion_factors = [1, 2]

class RISCVVectorPermutationVectorVectorScalar(RISCVVectorVectorVectorScalar):
    pass




class RISCVVectorVectorVectorImmediate(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <imm><vm>"
    inputs = ["Vb"]
    outputs = ["Vd"]

class RISCVVectorIntVectorVectorImmediate(RISCVVectorVectorVectorImmediate):
    pass

class RISCVVectorIntVectorVectorImmediateNarrowing(RISCVVectorIntVectorVectorImmediate):
    input_local_expansion_factors = [2]

class RISCVVectorFixedVectorVectorImmediate(RISCVVectorVectorVectorImmediate):
    pass

class RISCVVectorFixedVectorVectorImmediateNarrowing(RISCVVectorFixedVectorVectorImmediate):
    input_local_expansion_factors = [2]

class RISCVVectorPermutationVectorVectorImmediate(RISCVVectorVectorVectorImmediate):
    pass



class RISCVVectorIntVectorVector(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <Va><vm>"
    inputs = ["Va"]
    outputs = ["Vd"]




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


class RISCVVectorCompareVectorVector(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <Va>, <Vc>"
    inputs = ["Va", "Vb", "Vc"]
    outputs = ["Vd"]
    input_local_expansion_factors = [1, 1, 0]
    output_local_expansion_factors = [0]

class RISCVVectorCompareVectorScalar(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <Xa>, <Vc>"
    inputs = ["Xa", "Vb", "Vc"]
    outputs = ["Vd"]
    input_local_expansion_factors = [1, 1, 0]
    output_local_expansion_factors = [0]

class RISCVVectorCompareVectorImmediate(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <imm>, <Vc>"
    inputs = ["Vb", "Vc"]
    outputs = ["Vd"]
    input_local_expansion_factors = [1, 0]
    output_local_expansion_factors = [0]


class RISCVVectorLoad(RISCVVectorInstruction):
    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src)
        obj.increment = None
        obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]
        return obj

class RISCVVectorUnitStrideLoad(RISCVVectorLoad):
    pattern = "mnemonic <Vd>, (<Xa>)<vm>"
    inputs = ["Xa"]
    outputs = ["Vd"]

    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.output_local_expansion_factors = [float(obj.len) / obj.sew_external]

        return _expand_vector_registers_generic(obj)

class RISCVVectorUnitStrideMaskLoad(RISCVVectorLoad):
    pattern = "mnemonic <Vd>, (<Xa>)"
    inputs = ["Xa"]
    outputs = ["Vd"]

    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.output_local_expansion_factors = [float(obj.len) / obj.sew_external]

        return _expand_vector_registers_generic(obj)

class RISCVVectorStrideLoad(RISCVVectorLoad):
    pattern = "mnemonic <Vd>, (<Xa>), <Xb><vm>"
    inputs = ["Xa", "Xb"]
    outputs = ["Vd"]

    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.output_local_expansion_factors = [float(obj.len) / obj.sew_external]

        return _expand_vector_registers_generic(obj)

class RISCVVectorIndexedLoad(RISCVVectorLoad):
    pattern = "mnemonic <Vd>, (<Xa>), <Vb><vm>"
    inputs = ["Xa", "Vb"]
    outputs = ["Vd"]

    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.input_local_expansion_factors = [1, float(obj.len) / obj.sew_external]

        return _expand_vector_registers_generic(obj)

class RISCVVectorSegmentLoad(RISCVVectorLoad):
    pattern = "mnemonic <Vd>, (<Xa>)<vm>"
    inputs = ["Xa"]
    outputs = ["Vd"]

    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.output_local_expansion_factors = [(float(obj.len) / obj.sew_external) * obj.nf]

        return _expand_vector_registers_generic(obj)

class RISCVVectorStrideSegmentLoad(RISCVVectorLoad):
    pattern = "mnemonic <Vd>, (<Xa>), <Xb><vm>"
    inputs = ["Xa", "Xb"]
    outputs = ["Vd"]

    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.output_local_expansion_factors = [(float(obj.len) / obj.sew_external) * obj.nf]

        return _expand_vector_registers_generic(obj)

class RISCVVectorIndexedSegmentLoad(RISCVVectorLoad):
    pattern = "mnemonic <Vd>, (<Xa>), <Vb><vm>"
    inputs = ["Xa", "Vb"]
    outputs = ["Vd"]

    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.input_local_expansion_factors = [1, float(obj.len) / obj.sew_external]
        obj.output_local_expansion_factors = [obj.nf]

        return _expand_vector_registers_generic(obj)

class RISCVVectorWholeVectorLoad(RISCVVectorLoad):
    pattern = "mnemonic <Vd>, (<Xa>)"
    inputs = ["Xa"]
    outputs = ["Vd"]

    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.output_local_expansion_factors = [float(obj.nf) / obj.lmul_external]

        obj.increment = None
        #obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]

        return _expand_vector_registers_generic(obj)

class RISCVVectorStore(RISCVVectorInstruction):
    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src)
        obj.increment = None
        #obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]
        return obj

class RISCVVectorUnitStrideStore(RISCVVectorStore):
    pattern = "mnemonic <Va>, (<Xa>)<vm>"
    inputs = ["Xa", "Va"]
    outputs = []

    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.input_local_expansion_factors = [1, float(obj.len) / obj.sew_external]

        obj.increment = None
        #obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]

        return _expand_vector_registers_generic(obj)

class RISCVVectorUnitStrideMaskStore(RISCVVectorStore):
    pattern = "mnemonic <Va>, (<Xa>)"
    inputs = ["Xa", "Va"]
    outputs = []

    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.input_local_expansion_factors = [1, float(obj.len) / obj.sew_external]

        obj.increment = None
        #obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]

        return _expand_vector_registers_generic(obj)

class RISCVVectorStrideStore(RISCVVectorStore):
    pattern = "mnemonic <Va>, (<Xa>), <Xb><vm>"
    inputs = ["Xa", "Va", "Xb"]

    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.input_local_expansion_factors = [1, float(obj.len) / obj.sew_external, 1]

        obj.increment = None
        #obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]

        return _expand_vector_registers_generic(obj)

class RISCVVectorIndexedStore(RISCVVectorStore):
    pattern = "mnemonic <Va>, (<Xa>), <Vb><vm>"
    inputs = ["Xa", "Va", "Vb"]

    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.input_local_expansion_factors = [1, 1, float(obj.len) / obj.sew_external]

        obj.increment = None
        #obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]

        return _expand_vector_registers_generic(obj)


class RISCVVectorSegmentStore(RISCVVectorStore):
    pattern = "mnemonic <Vd>, (<Xa>)<vm>"
    inputs = ["Xa", "Vd"]

    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.input_local_expansion_factors = [1, (float(obj.len) / obj.sew_external) * obj.nf]

        obj.increment = None
        #obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]

        return _expand_vector_registers_generic(obj)

class RISCVVectorStrideSegmentStore(RISCVVectorStore):
    pattern = "mnemonic <Va>, (<Xa>), <Xb><vm>"
    inputs = ["Xa", "Va", "Xb"]

    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.input_local_expansion_factors = [1, (float(obj.len) / obj.sew_external) * obj.nf, 1]

        obj.increment = None
        #obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]

        return _expand_vector_registers_generic(obj)

class RISCVVectorIndexedSegmentStore(RISCVVectorStore):
    pattern = "mnemonic <Va>, (<Xa>), <Vb><vm>"
    inputs = ["Xa", "Va", "Vb"]

    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.input_local_expansion_factors = [1, obj.nf, float(obj.len) / obj.sew_external]

        obj.increment = None
        #obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]

        return _expand_vector_registers_generic(obj)

class RISCVVectorWholeVectorStore(RISCVVectorStore):
    pattern = "mnemonic <Va>, (<Xa>)"
    inputs = ["Xa", "Va"]

    @classmethod
    def make(cls, src):
        obj = RISCVVectorInstruction.build(cls, src, expand_registers=False)
        obj.input_local_expansion_factors = [1, float(obj.nf) / obj.lmul_external]

        obj.increment = None
        #obj.pre_index = obj.immediate
        obj.addr = obj.args_in[1]

        return _expand_vector_registers_generic(obj)


class RISCVVectorMoveScalarVector(RISCVVectorInstruction):
    pattern = "mnemonic <Xd>, <Va>"
    inputs = ["Va"]
    outputs = ["Xd"]

    @classmethod
    def make(cls, src):
        return RISCVVectorInstruction.build(cls, src, expand_registers=False)

class RISCVVectorMoveVectorScalar(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <Xa>"
    inputs = ["Xa"]
    outputs = ["Vd"]

    @classmethod
    def make(cls, src):
        return RISCVVectorInstruction.build(cls, src, expand_registers=False)

class RISCVVectorMoveVectorVector(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <Va>"
    inputs = ["Va"]
    outputs = ["Vd"]

    @classmethod
    def make(cls, src):
        return RISCVVectorInstruction.build(cls, src, expand_registers=False)

class RISCVVectorMoveVectorImmediate(RISCVVectorInstruction):
    pattern = "mnemonic <Vd>, <imm>"
    outputs = ["Vd"]

    @classmethod
    def make(cls, src):
        return RISCVVectorInstruction.build(cls, src, expand_registers=False)

v_instrs = [
    (["vsetvli"], vset_vl_i),
    (["vsetivli"], vset_i_vl_i),
    (["vsetvl"], vset_vl),

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
        RISCVVectorWholeVectorLoad
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
            "vs<nf>r.v",
            "vs<nf>re<len>.v"
        ],
        RISCVVectorWholeVectorStore
    ),

    (
        [
            "vmv.x.s",
            "vmv.x.v",
            #"vfmv.f.s"
        ],
        RISCVVectorMoveScalarVector
    ),
    (
        [
           "vmv.s.x",
            "vmv.v.x",
            #"vfmv.s.f"
        ],
        RISCVVectorMoveVectorScalar
    ),
    (
        [
            "vmv.v.v"
        ],
        RISCVVectorMoveVectorVector
    ),
    (
        [
            "vmv.v.i",
        ],
        RISCVVectorMoveVectorImmediate
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
            "vdivu.vv",
            "vdiv.vv",
            "vremu.vv",
            "vrem.vv",
            "vmulhu.vv",
            "vmulhsu.vv",
            "vsll.vv",
            "vmul.vv",
            "vmulh.vv",
            "vsrl.vv",
            "vsra.vv",

            "vcompress.vm",
        ],
        RISCVVectorIntVectorVectorVector,
    ),
    (
        [
            "vmadd.vv",
            "vnmsub.vv",
            "vmacc.vv",
            "vnmsac.vv",
        ],
        RISCVVectorIntVectorVectorVectorPassthrough
    ),
    (
        [
            "vwaddu.vv",
            "vwadd.vv",
            "vwsubu.vv",
            "vwsub.vv",
            "vwmulu.vv",
            "vwmulsu.vv",
            "vwmul.vv",
        ],
        RISCVVectorIntVectorVectorVectorWidening
    ),
    (
        [
            "vwaddu.wv",
            "vwadd.wv",
            "vwsubu.wv",
            "vwsub.wv",
        ],
        RISCVVectorIntVectorVectorVectorWideningVs2,
    ),
    (
        [
            "vwmaccu.vv",
            "vwmacc.vv",
            "vwmaccsu.vv",
        ],
        RISCVVectorIntVectorVectorVectorWideningPassthrough,
    ),
    (
        [
            "vnsrl.wv",
            "vnsra.wv",
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
            "vnclipu.wv",
            "vnclip.wv",
        ],
        RISCVVectorFixedVectorVectorVectorNarrowing
    ),
    (
        [
            "vrgather.vv",
        ],
        RISCVVectorPermutationVectorVectorVector,
    ),
    (
        [
            "vrgatherei16.vv",
        ],
        RISCVVectorPermutationVectorVectorVectorGatherE16
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
            "vdivu.vx",
            "vdiv.vx",
            "vremu.vx",
            "vrem.vx",
            "vmulhu.vx",
            "vmulhsu.vx",
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
            "vmadd.vx",
            "vnmsub.vx",
            "vmacc.vx",
            "vnmsac.vx",
        ],
        RISCVVectorIntVectorVectorScalarPassthrough,
    ),
    (
        [
            "vnsrl.wx",
            "vnsra.wx",
        ],
        RISCVVectorIntVectorVectorScalarNarrowing
    ),
    (
        [
            "vwaddu.vx",
            "vwadd.vx",
            "vwsubu.vx",
            "vwsub.vx",
            "vwmulu.vx",
            "vwmulsu.vx",
            "vwmul.vx",
        ],
        RISCVVectorIntVectorVectorScalarWidening,
    ),
    (
        [
            "vwaddu.wx",
            "vwadd.wx",
            "vwsubu.wx",
            "vwsub.wx",
        ],
        RISCVVectorIntVectorVectorScalarWideningVs2,
    ),
    (
        [
            "vwmaccu.vx",
            "vwmacc.vx",
            "vwmaccus.vx",
            "vwmaccsu.vx",
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
            "vnclipu.wx",
            "vnclip.wx",
        ],
        RISCVVectorFixedVectorVectorScalarNarrowing
    ),


    (
        [
            "vrgather.vx",
            "vslideup.vx",
            "vslide1up.vx",
            "vslide1down.vx",
            "vslidedown.vx",
            #"vfslide1up.vf",
            #"vfslide1down.vf",
        ],
        RISCVVectorPermutationVectorVectorScalar,
    ),


    (
        [
            # Vector Integer
            "vadd.vi",
            "vrsub.vi",
            "vand.vi",
            "vor.vi",
            "vxor.vi",
            "vsll.vi",
            "vsrl.vi",
            "vsra.vi",
        ],
        RISCVVectorIntVectorVectorImmediate,
    ),
    (
        [
            "vnsrl.wi",
            "vnsra.wi",
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
            "vnclipu.wi",
            "vnclip.wi",
        ],
        RISCVVectorFixedVectorVectorImmediateNarrowing,
    ),
    (
        [
            "vrgather.vi",
            "vslideup.vi",
            "vslidedown.vi",
            "vslide1down.vi",
        ],
        RISCVVectorPermutationVectorVectorImmediate,
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
            "vmandn.mm",
            "vmand.mm",
            "vmor.mm",
            "vmxor.mm",
            "vmorn.mm",
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
            "vmerge.vvm",
        ],
        RISCVVectorIntVectorVectorMask,
    ),
    (
        [
            "vadc.vxm",
            "vmadc.vxm",
            "vsbc.vxm",
            "vmsbc.vxm",
            "vmerge.vxm",
        ],
        RISCVVectorIntVectorScalarMask,
    ),
    (
        [
            "vadc.vim",
            "vmadc.vim",
            "vmerge.vim"
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
    (
        [
            "vnot.v"
        ],
        RISCVVectorIntVectorVector
    ),

    (
        [
            "vmseq.vv",
            "vmsne.vv",
            "vmsltu.vv",
            "vmslt.vv",
            "vmsleu.vv",
            "vmsle.vv",
        ],
        RISCVVectorCompareVectorVector
    ),
    (

        [
            "vmseq.vx",
            "vmsne.vx",
            "vmsltu.vx",
            "vmslt.vx",
            "vmsleu.vx",
            "vmsle.vx",
            "vmsgtu.vx",
            "vmsgt.vx",
        ],
        RISCVVectorCompareVectorScalar
    ),
    (
        [
            "vmseq.vi",
            "vmsne.vi",
            "vmsleu.vi",
            "vmsle.vi",
            "vmsgtu.vi",
            "vmsgt.vi",
        ],
        RISCVVectorCompareVectorImmediate
    )
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