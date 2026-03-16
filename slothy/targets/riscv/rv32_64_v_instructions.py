from slothy.targets.riscv.riscv_instruction_core import RISCVInstruction

# TODO: Expand inputs and outputs for LMUL

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



class RISCVVectorVectorVector(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <Va><vm>"
    inputs = ["Va", "Vb"]
    outputs = ["Vd"]

class RISCVVectorIntVectorVector(RISCVVectorVectorVector):
    pass

class RISCVVectorFixedVectorVector(RISCVVectorVectorVector):
    pass



class RISCVVectorVectorScalar(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <Xa><vm>"
    inputs = ["Vb", "Xa"]
    outputs = ["Vd"]

class RISCVVectorIntVectorScalar(RISCVVectorVectorScalar):
    pass

class RISCVVectorFixedVectorScalar(RISCVVectorVectorScalar):
    pass



class RISCVVectorVectorImmediate(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Va>, <imm><vm>"
    inputs = ["Va"]
    outputs = ["Vd"]

class RISCVVectorIntVectorImmediate(RISCVVectorVectorImmediate):
    pass

class RISCVVectorFixedVectorImmediate(RISCVVectorVectorImmediate):
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



class RISCVVectorFixedVectorVector(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <Va><vm>"
    inputs = ["Va", "Vb"]
    outputs = ["Vd"]

class RISCVVectorFixedVectorScalar(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Vb>, <Xa><vm>"
    inputs = ["Xa", "Vb"]
    outputs = ["Vd"]



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
            "vmadd.vv",
            "vnmsub.vv",
            "vnsrl.wv",
            "vnsra.wv",
            "vmacc.vv",
            "vnmsac.vv",
            "vwaddu.vv",
            "vwadd.vv",
            "vwsubu.vv",
            "vwsub.vv",
            "vwaddu.wv",
            "vwadd.wv",
            "vwsubu.wv",
            "vwsub.wv",
            "vwmulu.vv",
            "vwmulsu.vv",
            "vwmul.vv",
            "vwmaccu.vv",
            "vwmacc.vv",
            "vwmaccsu.vv",
        ],
        RISCVVectorIntVectorVector,
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
            "vnclipu.wv",
            "vnclip.wv",
        ],
        RISCVVectorFixedVectorVector,
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
            "vmadd.vx",
            "vnmsub.vx",
            "vnsrl.wx",
            "vnsra.wx",
            "vmacc.vx",
            "vnmsac.vx",
            "vwaddu.vx",
            "vwadd.vx",
            "vwsubu.vx",
            "vwsub.vx",
            "vwaddu.wx",
            "vwadd.wx",
            "vwsubu.wx",
            "vwsub.wx",
            "vwmulu.vx",
            "vwmulsu.vx",
            "vwmul.vx",
            "vwmaccu.vx",
            "vwmacc.vx",
            "vwmaccus.vx",
            "vwmaccsu.vx",
        ],
        RISCVVectorIntVectorScalar,
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
            "vnclipu.wx",
            "vnclip.wx",
        ],
        RISCVVectorFixedVectorScalar,
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
            "vnsrl.wi",
            "vnsra.wi",
        ],
        RISCVVectorIntVectorImmediate,
    ),
    (
        [
            "vsaddu.vi",
            "vsadd.vi",
            "vssrl.vi",
            "vssra.vi",
            "vnclipu.wi",
            "vnclip.wi",
        ],
        RISCVVectorFixedVectorImmediate,
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
            "vzext.vf<len>",
             "vsext.vf<len>",
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