from slothy.targets.riscv.riscv_instruction_core import RISCVInstruction

# TODO: Expand inputs and outputs for LMUL
# TODO: Add v0 as an input if the mask is selected

class v_set_vl_i(RISCVInstruction):
    pattern = "vsetvli <Xd>, <Xa>, <vtype>"
    inputs = ["Xa"]
    outputs = ["Xd", "vtype"] # TODO: Model vtype in output

class v_i_set_vl_i(RISCVInstruction):
    pattern = "vsetivli <Xd>, <imm>, <vtype>"
    outputs = ["Xd", "vtype"] # TODO: Model vtype in output

class v_set_vl(RISCVInstruction):
    pattern = "vsetvl <Xd>, <Xa>, <Xb>"
    inputs = ["Xa", "Xb"]
    outputs = ["Xd", "vtype"] # TODO: Model vtype in output



class RISCVVectorVectorVectorVector(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <Va><vm>"
    inputs = ["Va", "Vb"]
    outputs = ["Vd"]

class RISCVVectorIntVectorVectorVector(RISCVVectorVectorVectorVector):
    pass

class RISCVVectorIntVectorVectorVectorPassthrough(RISCVVectorVectorVectorVector):
    # TODO: Handle Vd being used as an input
    pass

class RISCVVectorIntVectorVectorVectorNarrowing(RISCVVectorIntVectorVectorVector):
    # TODO: Handle vs2 as 2*SEW
    pass

class RISCVVectorIntVectorVectorVectorWidening(RISCVVectorIntVectorVectorVector):
    # TODO: Handle the EMUL being twice the LMUL
    pass

class RISCVVectorIntVectorVectorVectorWideningPassthrough(RISCVVectorIntVectorVectorVector):
    # TODO: Handle Vd being 2*SEW in input and output
    # TODO: Handle Vd also being an input
    pass

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



class RISCVVectorVectorVectorScalar(RISCVInstruction):
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
    # TODO: Handle Vd being used as an input
    pass

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
    # TODO: Handle Vd also being an input
    pass



class RISCVVectorVectorVectorImmediate(RISCVInstruction):
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



class RISCVVectorIntVectorVectorMask(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <Va>, <Ve>" # Ve is mask (so is v0)
    inputs = ["Va", "Vb", "Ve"]
    outputs = ["Vd"]

class RISCVVectorIntVectorScalarMask(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <Xa>, <Ve>" # Ve is mask (so is v0)
    inputs = ["Va", "Vb", "Ve"]

class RISCVVectorIntVectorImmediateMask(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <imm>, <Ve>" # Ve is mask (so is v0)
    inputs = ["Vb", "Ve"]

class RISCVVectorIntVectorMask(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Vb><vm>"
    inputs = ["Vb"]
    output = ["Vd"]



class RISCVVectorMaskScalarVector(RISCVInstruction):
    pattern = "mnemonic <Xd>, <Vb><vm>"
    inputs = ["Vb"]
    outputs = ["Xd"]

class RISCVVectorMaskVectorVector(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Vb><vm>"
    inputs = ["Vb"]
    outputs = ["Vd"]

class RISCVVectorMaskVector(RISCVInstruction):
    pattern = "mnemonic <Vd><vm>"
    in_outs = ["Vd"]



class RISCVVectorUnitStrideLoad(RISCVInstruction):
    pattern = "mnemonic <Vd>, (<Xa>)<vm>"
    inputs = ["Xa", "vm"]
    outputs = ["Vd"]

class RISCVVectorUnitStrideMaskLoad(RISCVInstruction):
    pattern = "mnemonic <Vd>, (<Xa>)"
    inputs = ["Xa"]
    outputs = ["Vd"]

class RISCVVectorStrideLoad(RISCVInstruction):
    pattern = "mnemonic <Vd>, (<Xa>), <Xb><vm>"
    inputs = ["Xa", "Xb"]
    outputs = ["Vd"]

class RISCVVectorIndexedLoad(RISCVInstruction):
    pattern = "mnemonic <Vd>, (<Xa>), <Vb><vm>"
    inputs = ["Xa", "Vb"]
    outputs = ["Vd"]

class RISCVVectorIndexedSegmentLoad(RISCVInstruction):
    pattern = "mnemonic <Vd>, (<Xa>), <Xb><vm>"
    inputs = ["Xa", "Xb"]
    outputs = ["Vd"]

class RISCVWholeVectorLoad(RISCVInstruction):
    pattern = "mnemonic <Vd>, (<Xa>)"
    inputs = ["Xa"]
    outputs = ["Vd"]




class RISCVVectorUnitStrideStore(RISCVInstruction):
    pattern = "mnemonic <Va>, (<Xa>)vm"
    inputs = ["Va", "Xa"]
    outputs = []

class RISCVVectorUnitStrideMaskStore(RISCVInstruction):
    pattern = "mnemonic <Va>, (<Xa>)"
    inputs = ["Va", "Xa"]
    outputs = []

class RISCVVectorStrideStore(RISCVInstruction):
    pattern = "mnemonic <Va>, (<Xa>), <Xb><vm>"
    inputs = ["Va", "Xa", "Xb"]

class RISCVVectorIndexedStore(RISCVInstruction):
    pattern = "mnemonic <Va>, (<Xa>), <Xb><vm>"
    inputs = ["Va", "Xa", "Xb"]

class RISCVVectorIndexedSegmentStore(RISCVInstruction):
    pattern = "mnemonic <Va>, (<Xa>), <Vb><vm>"
    inputs = ["Va", "Xa", "Vb"]

class RISCVVectorWholeVectorStore(RISCVInstruction):
    pattern = "mnemonic <Va>, (<Xa>)"
    inputs = ["Va", "Xa"]


v_instrs = [
    (["vsetvli"], v_set_vl_i),
    (["visetvli"], v_i_set_vl_i),
    (["vsetvl"], v_set_vl),

    # Vector Load
    (["vle<len>.v", "vle<len>ff.v", "vlseg<nf>e<len>.v"], RISCVVectorUnitStrideLoad),
    (["vlm.v"], RISCVVectorUnitStrideMaskLoad),
    (["vluxei<len>.v", "vloxei<len>.v"], RISCVVectorIndexedLoad),
    (["vlse<len>.v", "vlsseg<nf>e<len>.v"], RISCVVectorStrideLoad),
    (["vluxseg<nf>ei<len>.v", "vloxseg<nf>ei<len>.v"], RISCVVectorIndexedSegmentLoad),
    (["vl<nf>re<len>.v", "vl<nf>r.v"], RISCVWholeVectorLoad),

    # Vector Store
    (["vse<len>.v", "vsseg<nf>e<len>.v"], RISCVVectorUnitStrideStore),
    (["vsm.v"], RISCVVectorUnitStrideMaskStore),
    (["vsse<len>.v", "vssseg<nf>e<len>.v"], RISCVVectorStrideStore),
    (["vsuxei<len>.v", "vsoxei<len>.v"], RISCVVectorIndexedStore),
    (["vsuxseg<nf>ei<len>.v", "vsoxseg<nf>ei<len>.v"], RISCVVectorIndexedSegmentStore),
    (["vs<nf>r.v"], RISCVWholeVectorLoad),

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