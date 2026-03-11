from slothy.targets.riscv.riscv_instruction_core import RISCVInstruction


class v_set_vl_i(RISCVInstruction):
    pattern = "vsetvli <Xd>, <Xa>, <T>, <A>, <E>, <M>, <Tail>, <Mask>"
    inputs = ["Xa"]
    outputs = ["Xd"] # TODO: Model vtype in output

class v_i_set_vl_i(RISCVInstruction):
    pattern = "vsetivli <Xd>, <imm>, <T>, <A>, <E>, <M>, <Tail>, <Mask>"
    outputs = ["Xd"] # TODO: Model vtype in output

class v_set_vl(RISCVInstruction):
    pattern = "vsetvl <Xd>, <Xa>, <Xb>"
    inputs = ["Xa", "Xb"]
    outputs = ["Xd"] # TODO: Model vtype in output