# flake8: noqa: F405
#
# Copyright (c) 2022 Arm Limited
# Copyright (c) 2022 Hanno Becker
# Copyright (c) 2023 Amin Abdulrahman, Matthias Kannwischer
# Copyright (c) 2024 Justus Bergermann
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Authors: Hanno Becker <hannobecker@posteo.de>
#          Justus Bergermann <mail@justus-bergermann.de>
#

"""
Experimental XuanTie C908 microarchitecture model for SLOTHY

Some data in this model is derived from the XuanTie C908 manual, other is derived from microbenchmarks.

WARNING: The data in this module is approximate and may contain errors.
"""

################################### NOTE ###############################################  # noqa: E266
###                                                                                  ###  # noqa: E266
### WARNING: The data in this module is approximate and may contain errors.          ###  # noqa: E266
###          They are _NOT_ an official software optimization guide for C908.        ###  # noqa: E266
###                                                                                  ###  # noqa: E266
########################################################################################  # noqa: E266

from enum import Enum
from unittest import case

from slothy.targets.riscv.riscv import *  # noqa: F403
from slothy.targets.riscv.rv32_64_i_instructions import *  # noqa: F403
from slothy.targets.riscv.rv32_64_m_instructions import *  # noqa: F403
from slothy.targets.riscv.rv32_64_b_instructions import *  # noqa: F403
from slothy.targets.riscv.rv32_64_pseudo_instructions import *  # noqa: F403
from slothy.targets.riscv.rv32_64_v_instructions import RISCVVectorSetVtype, RISCVVectorInstruction

issue_rate = 2
llvm_mca_target = ""


class ExecutionUnit(Enum):
    """Enumeration of execution units in C908 model"""

    SCALAR_ALU0 = 1
    SCALAR_ALU1 = 2
    SCALAR_MUL = 3
    LSU = 4
    VEC0 = 5
    VEC1 = 6

    def __repr__(self):
        return self.name

    @classmethod
    def SCALAR(cls):  # pylint: disable=invalid-name
        """All scalar execution units"""
        return [ExecutionUnit.SCALAR_ALU0, ExecutionUnit.SCALAR_ALU1]


# Opaque function called by SLOTHY to add further microarchitecture-
# specific constraints which are not encapsulated by the general framework.
def add_further_constraints(slothy):
    if slothy.config.constraints.functional_only:
        return


# Opaque function called by SLOTHY to add further microarchitecture-
# specific objectives.
def has_min_max_objective(config):
    """Adds C908-"""
    _ = config
    return False


def get_min_max_objective(slothy):
    _ = slothy
    return


execution_units = {
    (  # TODO: use existing instructions list or superclass
        RISCVInstruction.classes_by_names["addi"],
        RISCVInstruction.classes_by_names["slti"],
        RISCVInstruction.classes_by_names["sltiu"],
        RISCVInstruction.classes_by_names["andi"],
        RISCVInstruction.classes_by_names["ori"],
        RISCVInstruction.classes_by_names["xori"],
        RISCVInstruction.classes_by_names["slli"],
        RISCVInstruction.classes_by_names["srli"],
        RISCVInstruction.classes_by_names["srai"],
        RISCVInstruction.classes_by_names["andcls"],
        RISCVInstruction.classes_by_names["orcls"],
        RISCVInstruction.classes_by_names["xor"],
        RISCVInstruction.classes_by_names["add"],
        RISCVInstruction.classes_by_names["slt"],
        RISCVInstruction.classes_by_names["sltu"],
        RISCVInstruction.classes_by_names["sll"],
        RISCVInstruction.classes_by_names["srl"],
        RISCVInstruction.classes_by_names["sub"],
        RISCVInstruction.classes_by_names["sra"],
        RISCVInstruction.classes_by_names["lui"],
        RISCVInstruction.classes_by_names["auipc"],
        # Zbkb extension - Bit manipulation for cryptography
        # TODO: Verify performance characteristics for C908
        RISCVInstruction.classes_by_names["rol"],
        RISCVInstruction.classes_by_names["ror"],
        RISCVInstruction.classes_by_names["rori"],
        RISCVInstruction.classes_by_names["andn"],
        RISCVInstruction.classes_by_names["orn"],
        RISCVInstruction.classes_by_names["xnor"],
        RISCVInstruction.classes_by_names["pack"],
        RISCVInstruction.classes_by_names["packh"],
        RISCVInstruction.classes_by_names["brev8"],
        RISCVInstruction.classes_by_names["rev8"],
        RISCVInstruction.classes_by_names["zip"],
        RISCVInstruction.classes_by_names["unzip"],
        # Pseudo-instructions
        RISCVInstruction.classes_by_names["li"],
        RISCVInstruction.classes_by_names["mv"],
        RISCVInstruction.classes_by_names["neg"],
        RISCVInstruction.classes_by_names["not"],
        RISCVInstruction.classes_by_names["la"],
    ): ExecutionUnit.SCALAR(),
    (
        RISCVInstruction.classes_by_names["lb"],
        RISCVInstruction.classes_by_names["lbu"],
        RISCVInstruction.classes_by_names["lh"],
        RISCVInstruction.classes_by_names["lhu"],
        RISCVInstruction.classes_by_names["lw"],
        RISCVInstruction.classes_by_names["lwu"],
        RISCVInstruction.classes_by_names["ld"],
        RISCVInstruction.classes_by_names["sb"],
        RISCVInstruction.classes_by_names["sh"],
        RISCVInstruction.classes_by_names["sw"],
        RISCVInstruction.classes_by_names["sd"],
    ): ExecutionUnit.LSU,
    (
        RISCVInstruction.classes_by_names["mul"],
        RISCVInstruction.classes_by_names["mulh"],
        RISCVInstruction.classes_by_names["mulhsu"],
        RISCVInstruction.classes_by_names["mulhu"],
        RISCVInstruction.classes_by_names["div"],
        RISCVInstruction.classes_by_names["divu"],
        RISCVInstruction.classes_by_names["rem"],
        RISCVInstruction.classes_by_names["remu"],
    ): ExecutionUnit.SCALAR_MUL,

    # Mainly educated guesses
    (
        RISCVInstruction.classes_by_names["vmandn.mm"],
        RISCVInstruction.classes_by_names["vmand.mm"],
        RISCVInstruction.classes_by_names["vmor.mm"],
        RISCVInstruction.classes_by_names["vmxor.mm"],
        RISCVInstruction.classes_by_names["vmorn.mm"],
        RISCVInstruction.classes_by_names["vmnand.mm"],
        RISCVInstruction.classes_by_names["vmnor.mm"],
        RISCVInstruction.classes_by_names["vmxnor.mm"],

        RISCVInstruction.classes_by_names["vadd.vv"],
        RISCVInstruction.classes_by_names["vadd.vx"],
        RISCVInstruction.classes_by_names["vadd.vi"],
        RISCVInstruction.classes_by_names["vsub.vv"],
        RISCVInstruction.classes_by_names["vsub.vx"],
        RISCVInstruction.classes_by_names["vrsub.vx"],
        RISCVInstruction.classes_by_names["vrsub.vi"],
        RISCVInstruction.classes_by_names["vminu.vx"],
        RISCVInstruction.classes_by_names["vmin.vx"],
        RISCVInstruction.classes_by_names["vmaxu.vx"],
        RISCVInstruction.classes_by_names["vmax.vx"],
        RISCVInstruction.classes_by_names["vminu.vv"],
        RISCVInstruction.classes_by_names["vmin.vv"],
        RISCVInstruction.classes_by_names["vmaxu.vv"],
        RISCVInstruction.classes_by_names["vmax.vv"],

        RISCVInstruction.classes_by_names["vaaddu.vv"],
        RISCVInstruction.classes_by_names["vaaddu.vx"],
        RISCVInstruction.classes_by_names["vaadd.vv"],
        RISCVInstruction.classes_by_names["vaadd.vx"],

        RISCVInstruction.classes_by_names["vasubu.vv"],
        RISCVInstruction.classes_by_names["vasubu.vx"],
        RISCVInstruction.classes_by_names["vasub.vv"],
        RISCVInstruction.classes_by_names["vasub.vx"],

        RISCVInstruction.classes_by_names["vsaddu.vv"],
        RISCVInstruction.classes_by_names["vsaddu.vx"],
        RISCVInstruction.classes_by_names["vsaddu.vi"],
        RISCVInstruction.classes_by_names["vsadd.vv"],
        RISCVInstruction.classes_by_names["vsadd.vx"],
        RISCVInstruction.classes_by_names["vsadd.vi"],

        RISCVInstruction.classes_by_names["vssubu.vv"],
        RISCVInstruction.classes_by_names["vssubu.vx"],
        RISCVInstruction.classes_by_names["vssub.vv"],
        RISCVInstruction.classes_by_names["vssub.vx"],

        RISCVInstruction.classes_by_names["vsmul.vv"],
        RISCVInstruction.classes_by_names["vsmul.vx"],

        # Not exact, but close enough
        RISCVInstruction.classes_by_names["vnmsub.vv"],
        RISCVInstruction.classes_by_names["vnmsub.vx"],
        RISCVInstruction.classes_by_names["vnmsac.vv"],
        RISCVInstruction.classes_by_names["vnmsac.vx"],

        RISCVInstruction.classes_by_names["vmsbf.m"],
        RISCVInstruction.classes_by_names["vmsof.m"],
        RISCVInstruction.classes_by_names["vmsif.m"],

        RISCVInstruction.classes_by_names["vmulhu.vv"],
        RISCVInstruction.classes_by_names["vmulhu.vx"],
        RISCVInstruction.classes_by_names["vmul.vv"],
        RISCVInstruction.classes_by_names["vmul.vx"],
        RISCVInstruction.classes_by_names["vmulhsu.vv"],
        RISCVInstruction.classes_by_names["vmulhsu.vx"],
        RISCVInstruction.classes_by_names["vmulh.vv"],
        RISCVInstruction.classes_by_names["vmulh.vx"],
        RISCVInstruction.classes_by_names["vmadd.vv"],
        RISCVInstruction.classes_by_names["vmadd.vx"],
        RISCVInstruction.classes_by_names["vmacc.vv"],
        RISCVInstruction.classes_by_names["vmacc.vx"],
    ): [ExecutionUnit.VEC0, ExecutionUnit.VEC1],
    (
        RISCVInstruction.classes_by_names["vand.vv"],
        RISCVInstruction.classes_by_names["vand.vx"],
        RISCVInstruction.classes_by_names["vand.vi"],
        RISCVInstruction.classes_by_names["vor.vv"],
        RISCVInstruction.classes_by_names["vor.vx"],
        RISCVInstruction.classes_by_names["vor.vi"],
        RISCVInstruction.classes_by_names["vxor.vv"],
        RISCVInstruction.classes_by_names["vxor.vx"],
        RISCVInstruction.classes_by_names["vxor.vi"],
        RISCVInstruction.classes_by_names["vnot.v"],
        RISCVInstruction.classes_by_names["vrgather.vx"],
        RISCVInstruction.classes_by_names["vrgather.vi"],
        RISCVInstruction.classes_by_names["vslideup.vx"],
        RISCVInstruction.classes_by_names["vslideup.vi"],
        RISCVInstruction.classes_by_names["vslide1up.vx"],

        # Simplified
        RISCVInstruction.classes_by_names["vadc.vvm"],
        RISCVInstruction.classes_by_names["vadc.vxm"],
        RISCVInstruction.classes_by_names["vadc.vim"],
        RISCVInstruction.classes_by_names["vsbc.vvm"],
        RISCVInstruction.classes_by_names["vsbc.vxm"],

        RISCVInstruction.classes_by_names["vmerge.vvm"],
        RISCVInstruction.classes_by_names["vmerge.vxm"],
        RISCVInstruction.classes_by_names["vmerge.vim"],

        RISCVInstruction.classes_by_names["vmv.v.v"],
        RISCVInstruction.classes_by_names["vmv.v.x"],
        RISCVInstruction.classes_by_names["vmv.v.i"],

        RISCVInstruction.classes_by_names["vsll.vv"],
        RISCVInstruction.classes_by_names["vsll.vx"],
        RISCVInstruction.classes_by_names["vsll.vi"],

        RISCVInstruction.classes_by_names["vsrl.vv"],
        RISCVInstruction.classes_by_names["vsrl.vx"],
        RISCVInstruction.classes_by_names["vsrl.vi"],
        RISCVInstruction.classes_by_names["vsra.vv"],
        RISCVInstruction.classes_by_names["vsra.vx"],
        RISCVInstruction.classes_by_names["vsra.vi"],
        RISCVInstruction.classes_by_names["vssrl.vv"],
        RISCVInstruction.classes_by_names["vssrl.vx"],
        RISCVInstruction.classes_by_names["vssrl.vi"],

        RISCVInstruction.classes_by_names["vwaddu.vv"],
        RISCVInstruction.classes_by_names["vwaddu.vx"],
        RISCVInstruction.classes_by_names["vwadd.vv"],
        RISCVInstruction.classes_by_names["vwadd.vx"],
        RISCVInstruction.classes_by_names["vwsub.vv"],
        RISCVInstruction.classes_by_names["vwsub.vx"],
        RISCVInstruction.classes_by_names["vwaddu.wv"],
        RISCVInstruction.classes_by_names["vwaddu.wx"],
        RISCVInstruction.classes_by_names["vwadd.wv"],
        RISCVInstruction.classes_by_names["vwadd.wx"],
        RISCVInstruction.classes_by_names["vwsub.wv"],
        RISCVInstruction.classes_by_names["vwsub.wx"],
        RISCVInstruction.classes_by_names["vwmulu.vv"],
        RISCVInstruction.classes_by_names["vwmulu.vx"],
        RISCVInstruction.classes_by_names["vwmulsu.vv"],
        RISCVInstruction.classes_by_names["vwmul.vv"],
        RISCVInstruction.classes_by_names["vwmul.vx"],
        RISCVInstruction.classes_by_names["vwmaccu.vv"],
        RISCVInstruction.classes_by_names["vwmaccu.vx"],
        RISCVInstruction.classes_by_names["vwmacc.vv"],
        RISCVInstruction.classes_by_names["vwmacc.vx"],
        RISCVInstruction.classes_by_names["vwmaccsu.vv"],
        RISCVInstruction.classes_by_names["vwmaccsu.vx"],
        RISCVInstruction.classes_by_names["vwmaccus.vx"],

        RISCVInstruction.classes_by_names["vzext.vf"],
        RISCVInstruction.classes_by_names["vsext.vf"],

        RISCVInstruction.classes_by_names["viota.m"],
        RISCVInstruction.classes_by_names["vid.v"],

        RISCVInstruction.classes_by_names["vnsrl.wv"],
        RISCVInstruction.classes_by_names["vnsrl.wx"],
        RISCVInstruction.classes_by_names["vnsrl.wi"],
        RISCVInstruction.classes_by_names["vnsra.wv"],
        RISCVInstruction.classes_by_names["vnsra.wx"],
        RISCVInstruction.classes_by_names["vnsra.wi"],
        RISCVInstruction.classes_by_names["vnclipu.wv"],
        RISCVInstruction.classes_by_names["vnclipu.wx"],
        RISCVInstruction.classes_by_names["vnclipu.wi"],

        RISCVInstruction.classes_by_names["vrgather.vv"],

        RISCVInstruction.classes_by_names["vslidedown.vx"],
        RISCVInstruction.classes_by_names["vslide1down.vx"],
        RISCVInstruction.classes_by_names["vslide1down.vi"],

        RISCVInstruction.classes_by_names["vmadc.vvm"],
        RISCVInstruction.classes_by_names["vmadc.vxm"],
        RISCVInstruction.classes_by_names["vmadc.vim"],
        RISCVInstruction.classes_by_names["vmsbc.vvm"],
        RISCVInstruction.classes_by_names["vmsbc.vxm"],

        RISCVInstruction.classes_by_names["vmseq.vv"],
        RISCVInstruction.classes_by_names["vmseq.vx"],
        RISCVInstruction.classes_by_names["vmseq.vi"],

        RISCVInstruction.classes_by_names["vmsne.vv"],
        RISCVInstruction.classes_by_names["vmsne.vx"],
        RISCVInstruction.classes_by_names["vmsne.vi"],

        RISCVInstruction.classes_by_names["vmsltu.vv"],
        RISCVInstruction.classes_by_names["vmsltu.vx"],
        RISCVInstruction.classes_by_names["vmslt.vv"],
        RISCVInstruction.classes_by_names["vmslt.vx"],
        RISCVInstruction.classes_by_names["vmsleu.vv"],
        RISCVInstruction.classes_by_names["vmsleu.vx"],
        RISCVInstruction.classes_by_names["vmsleu.vi"],
        RISCVInstruction.classes_by_names["vmsle.vv"],
        RISCVInstruction.classes_by_names["vmsle.vx"],
        RISCVInstruction.classes_by_names["vmsle.vi"],
        RISCVInstruction.classes_by_names["vmsgtu.vx"],
        RISCVInstruction.classes_by_names["vmsgtu.vi"],
        RISCVInstruction.classes_by_names["vmsgt.vx"],
        RISCVInstruction.classes_by_names["vmsgt.vi"],

        RISCVInstruction.classes_by_names["vle.v"],
        RISCVInstruction.classes_by_names["vleff.v"],
        RISCVInstruction.classes_by_names["vlse.v"],
        RISCVInstruction.classes_by_names["vl.v"],
        RISCVInstruction.classes_by_names["vlr.v"],
        RISCVInstruction.classes_by_names["vluxei.v"],
        RISCVInstruction.classes_by_names["vloxei.v"],
        RISCVInstruction.classes_by_names["vlseg.v"],
        RISCVInstruction.classes_by_names["vlsseg.v"],
        RISCVInstruction.classes_by_names["vluxseg.v"],
        RISCVInstruction.classes_by_names["vloxseg.v"],
        RISCVInstruction.classes_by_names["vse.v"],
        RISCVInstruction.classes_by_names["vsse.v"],
        RISCVInstruction.classes_by_names["vssseg.v"],
        RISCVInstruction.classes_by_names["vsseg.v"],
        RISCVInstruction.classes_by_names["vsuxseg.v"],
        RISCVInstruction.classes_by_names["vsoxseg.v"],
        RISCVInstruction.classes_by_names["vsuxei.v"],
        RISCVInstruction.classes_by_names["vsoxei.v"],
        RISCVInstruction.classes_by_names["vs.v"],
        RISCVInstruction.classes_by_names["vsr.v"],

        RISCVInstruction.classes_by_names["vsetvl"],
        RISCVInstruction.classes_by_names["vsetvli"],
        RISCVInstruction.classes_by_names["vsetivli"]
    ): ExecutionUnit.VEC0,
}

def vrgather_vv_inverse_throughput(obj):
    match obj.lmul_external:
        case 1: return 4
        case 2: return 16
        case 4: return 65
        case 8: return 262
        case _: raise Exception("No valid case for vrgather_vv_inverse_throughput")

def vrgather16_inverse_throughput(obj):
    match (obj.lmul_external, obj.sew_external):
        case (1, 8): return 8
        case (2, 8): return 33
        case (4, 8): return 131
        case (1, _): return 4
        case (2, _): return 16
        case (4, _): return 66
        case (8, _): return 262
        case _: raise Exception("No valid case for vrgather16_inverse_throughput")

def vslidedown_inverse_throughput(obj):
    match obj.lmul_external:
        case 1: return 3
        case 2: return 5
        case 4: return 9
        case 8: return 18
        case _: raise Exception("No valid case for vslidedown_inverse_throughput")

def vmasbc_inverse_throughput(obj):
    match (obj.lmul_external, obj.sew_external):
        case (1, _): return 2
        case (2, _): return 4
        case (4, _): return 8
        case (8, 8): return 11
        case (8, _): return 19
        case _: raise Exception("No valid case for vmasbc_inverse_throughput")

def vcompress_inverse_throughput(obj):
    match obj.lmul_external:
        case 1: return 3
        case 2: return 10
        case 4: return 37
        case 8: return 139
        case _: raise Exception("No valid case for vcompress_inverse_throughput")

def vmvr_inverse_throughput(obj):
    match (obj.lmul_external, obj.len):
        case (8, 1): return 4
        case (_, 1): return 2
        case (_, 2): return 4
        case (_, 4): return 8
        case (_, 8): return 16
        case _: raise Exception("No valid case for vmvr_inverse_throughput")

def vdivuvv_inverse_throughput(obj):
    match (obj.lmul_external, obj.sew_external):
        case (1, 8): return 26
        case (2, 8): return 52
        case (4, 8): return 104
        case (8, 8): return 206

        case (1, 16): return 23
        case (2, 16): return 46
        case (4, 16): return 92
        case (8, 16): return 185

        case (1, 32): return 22
        case (2, 32): return 44
        case (4, 32): return 87
        case (8, 32): return 174

        case (1, 64): return 21
        case (2, 64): return 42
        case (4, 64): return 84
        case (8, 64): return 166

        case _: raise Exception("No valid case for vdivuvv_inverse_throughput")

def vdivuvx_inverse_throughput(obj):
    match (obj.lmul_external, obj.sew_external):
        case (1, 8): return 21
        case (2, 8): return 41
        case (4, 8): return 83
        case (8, 8): return 165

        case (1, 16): return 21
        case (2, 16): return 42
        case (4, 16): return 86
        case (8, 16): return 164

        case (1, 32): return 21
        case (2, 32): return 41
        case (4, 32): return 83
        case (8, 32): return 166

        case (1, 64): return 25
        case (2, 64): return 47
        case (4, 64): return 106
        case (8, 64): return 164

        case _: raise Exception("No valid case for vdivuvx_inverse_throughput")

def vdivvv_inverse_throughput(obj):
    match (obj.lmul_external, obj.sew_external):
        case (1, 8): return 27
        case (2, 8): return 54
        case (4, 8): return 107
        case (8, 8): return 213

        case (1, 16): return 25
        case (2, 16): return 49
        case (4, 16): return 99
        case (8, 16): return 197

        case (1, 32): return 24
        case (2, 32): return 47
        case (4, 32): return 94
        case (8, 32): return 188

        case (1, 64): return 23
        case (2, 64): return 46
        case (4, 64): return 91
        case (8, 64): return 182

        case _: raise Exception("No valid case for vdivvv_inverse_throughput")

def vdivvx_inverse_throughput(obj):
    match (obj.lmul_external, obj.sew_external):
        case (1, 8): return 25
        case (2, 8): return 47
        case (4, 8): return 99
        case (8, 8): return 183

        case (1, 16): return 23
        case (2, 16): return 45
        case (4, 16): return 91
        case (8, 16): return 182

        case (1, 32): return 23
        case (2, 32): return 46
        case (4, 32): return 94
        case (8, 32): return 182

        case (1, 64): return 27
        case (2, 64): return 51
        case (4, 64): return 114
        case (8, 64): return 180

        case _: raise Exception("No valid case for vdivvx_inverse_throughput")


def vremuvv_inverse_throughput(obj):
    match (obj.lmul_external, obj.sew_external):
        case (1, 8): return 28
        case (2, 8): return 56
        case (4, 8): return 112
        case (8, 8): return 224

        case (1, 16): return 25
        case (2, 16): return 50
        case (4, 16): return 100
        case (8, 16): return 201

        case (1, 32): return 24
        case (2, 32): return 48
        case (4, 32): return 96
        case (8, 32): return 190

        case (1, 64): return 23
        case (2, 64): return 46
        case (4, 64): return 91
        case (8, 64): return 183

        case _: raise Exception("No valid case for vremuvv_inverse_throughput")

def vremuvx_inverse_throughput(obj):
    match (obj.lmul_external, obj.sew_external):
        case (1, 8): return 24
        case (2, 8): return 48
        case (4, 8): return 103
        case (8, 8): return 181

        case (1, 16): return 23
        case (2, 16): return 46
        case (4, 16): return 94
        case (8, 16): return 182

        case (1, 32): return 23
        case (2, 32): return 45
        case (4, 32): return 91
        case (8, 32): return 182

        case (1, 64): return 27
        case (2, 64): return 51
        case (4, 64): return 114
        case (8, 64): return 180

        case _: raise Exception("No valid case for vremuvx_inverse_throughput")

def vremvv_inverse_throughput(obj):
    match (obj.lmul_external, obj.sew_external):
        case (1, 8): return 29
        case (2, 8): return 58
        case (4, 8): return 115
        case (8, 8): return 231

        case (1, 16): return 27
        case (2, 16): return 54
        case (4, 16): return 107
        case (8, 16): return 213

        case (1, 32): return 26
        case (2, 32): return 51
        case (4, 32): return 102
        case (8, 32): return 205

        case (1, 64): return 25
        case (2, 64): return 50
        case (4, 64): return 99
        case (8, 64): return 198

        case _: raise Exception("No valid case for vremvv_inverse_throughput")

def vremvx_inverse_throughput(obj):
    match (obj.lmul_external, obj.sew_external):
        case (1, 8): return 27
        case (2, 8): return 52
        case (4, 8): return 107
        case (8, 8): return 199

        case (1, 16): return 25
        case (2, 16): return 50
        case (4, 16): return 99
        case (8, 16): return 198

        case (1, 32): return 25
        case (2, 32): return 50
        case (4, 32): return 102
        case (8, 32): return 199

        case (1, 64): return 29
        case (2, 64): return 55
        case (4, 64): return 122
        case (8, 64): return 196

        case _: raise Exception("No valid case for vremvx_inverse_throughput")

def lmul1_slow_sew64_inverse_throughput(obj):
    match (obj.lmul_external, obj.sew_external):
        case (1, 64): return 4
        case (2, 64): return 8
        case (4, 64): return 16
        case (8, 64): return 33
        case _: return obj.lmul_external

inverse_throughput = {
    (
        RISCVInstruction.classes_by_names["addi"],
        RISCVInstruction.classes_by_names["slti"],
        RISCVInstruction.classes_by_names["sltiu"],
        RISCVInstruction.classes_by_names["andi"],
        RISCVInstruction.classes_by_names["ori"],
        RISCVInstruction.classes_by_names["xori"],
        RISCVInstruction.classes_by_names["slli"],
        RISCVInstruction.classes_by_names["srli"],
        RISCVInstruction.classes_by_names["srai"],
        RISCVInstruction.classes_by_names["andcls"],
        RISCVInstruction.classes_by_names["orcls"],
        RISCVInstruction.classes_by_names["xor"],
        RISCVInstruction.classes_by_names["add"],
        RISCVInstruction.classes_by_names["slt"],
        RISCVInstruction.classes_by_names["sltu"],
        RISCVInstruction.classes_by_names["sll"],
        RISCVInstruction.classes_by_names["srl"],
        RISCVInstruction.classes_by_names["sub"],
        RISCVInstruction.classes_by_names["sra"],
        RISCVInstruction.classes_by_names["lui"],
        RISCVInstruction.classes_by_names["auipc"],
        # Zbkb extension - Bit manipulation for cryptography
        # TODO: Verify performance characteristics for C908
        RISCVInstruction.classes_by_names["rol"],
        RISCVInstruction.classes_by_names["ror"],
        RISCVInstruction.classes_by_names["rori"],
        RISCVInstruction.classes_by_names["andn"],
        RISCVInstruction.classes_by_names["orn"],
        RISCVInstruction.classes_by_names["xnor"],
        RISCVInstruction.classes_by_names["pack"],
        RISCVInstruction.classes_by_names["packh"],
        RISCVInstruction.classes_by_names["brev8"],
        RISCVInstruction.classes_by_names["rev8"],
        RISCVInstruction.classes_by_names["zip"],
        RISCVInstruction.classes_by_names["unzip"],
        # Pseudo-instructions
        RISCVInstruction.classes_by_names["li"],
        RISCVInstruction.classes_by_names["mv"],
        RISCVInstruction.classes_by_names["neg"],
        RISCVInstruction.classes_by_names["not"],
        RISCVInstruction.classes_by_names["la"],
    ): 1,
    (
        RISCVInstruction.classes_by_names["lb"],
        RISCVInstruction.classes_by_names["lbu"],
        RISCVInstruction.classes_by_names["lh"],
        RISCVInstruction.classes_by_names["lhu"],
        RISCVInstruction.classes_by_names["lw"],
        RISCVInstruction.classes_by_names["lwu"],
        RISCVInstruction.classes_by_names["ld"],
        RISCVInstruction.classes_by_names["sb"],
        RISCVInstruction.classes_by_names["sh"],
        RISCVInstruction.classes_by_names["sw"],
        RISCVInstruction.classes_by_names["sd"],
    ): 1,
    (
        RISCVInstruction.classes_by_names["mul"],
        RISCVInstruction.classes_by_names["mulh"],
        RISCVInstruction.classes_by_names["mulhsu"],
        RISCVInstruction.classes_by_names["mulhu"],
        RISCVInstruction.classes_by_names["div"],
        RISCVInstruction.classes_by_names["divu"],
        RISCVInstruction.classes_by_names["rem"],
        RISCVInstruction.classes_by_names["remu"],
    ): 2,

    # Vector Instructions (Thanks to https://camel-cdr.github.io/rvv-bench-results/canmv_k230/index.html :D)
    # TODO: Make timings account for masking

    (
        RISCVInstruction.classes_by_names["vsetvl"],
        RISCVInstruction.classes_by_names["vsetvli"],
        RISCVInstruction.classes_by_names["vsetivli"]
    ): 4,

    (
        RISCVInstruction.classes_by_names["vmandn.mm"],
        RISCVInstruction.classes_by_names["vmand.mm"],
        RISCVInstruction.classes_by_names["vmor.mm"],
        RISCVInstruction.classes_by_names["vmxor.mm"],
        RISCVInstruction.classes_by_names["vmorn.mm"],
        RISCVInstruction.classes_by_names["vmnand.mm"],
        RISCVInstruction.classes_by_names["vmnor.mm"],
        RISCVInstruction.classes_by_names["vmxnor.mm"],
    ): 1,

    (
        RISCVInstruction.classes_by_names["vadd.vv"], # Doubles for mask
        RISCVInstruction.classes_by_names["vadd.vx"],
        RISCVInstruction.classes_by_names["vadd.vi"],
        RISCVInstruction.classes_by_names["vsub.vv"],
        RISCVInstruction.classes_by_names["vsub.vx"],
        RISCVInstruction.classes_by_names["vrsub.vx"],
        RISCVInstruction.classes_by_names["vrsub.vi"],
        RISCVInstruction.classes_by_names["vminu.vx"], # Changes for mask
        RISCVInstruction.classes_by_names["vmin.vx"],  # Changes for mask
        RISCVInstruction.classes_by_names["vmaxu.vx"], # Changes for mask
        RISCVInstruction.classes_by_names["vmax.vx"],  # Changes for mask
        RISCVInstruction.classes_by_names["vminu.vv"], # Changes for mask
        RISCVInstruction.classes_by_names["vmin.vv"],  # Changes for mask
        RISCVInstruction.classes_by_names["vmaxu.vv"], # Changes for mask
        RISCVInstruction.classes_by_names["vmax.vv"],  # Changes for mask

        RISCVInstruction.classes_by_names["vaaddu.vv"],
        RISCVInstruction.classes_by_names["vaaddu.vx"],
        RISCVInstruction.classes_by_names["vaadd.vv"],
        RISCVInstruction.classes_by_names["vaadd.vx"],

        RISCVInstruction.classes_by_names["vasubu.vv"],
        RISCVInstruction.classes_by_names["vasubu.vx"],
        RISCVInstruction.classes_by_names["vasub.vv"],
        RISCVInstruction.classes_by_names["vasub.vx"],

        RISCVInstruction.classes_by_names["vsaddu.vv"],
        RISCVInstruction.classes_by_names["vsaddu.vx"],
        RISCVInstruction.classes_by_names["vsaddu.vi"],
        RISCVInstruction.classes_by_names["vsadd.vv"],
        RISCVInstruction.classes_by_names["vsadd.vx"],
        RISCVInstruction.classes_by_names["vsadd.vi"],

        RISCVInstruction.classes_by_names["vssubu.vv"],
        RISCVInstruction.classes_by_names["vssubu.vx"],
        RISCVInstruction.classes_by_names["vssub.vv"],
        RISCVInstruction.classes_by_names["vssub.vx"],

        RISCVInstruction.classes_by_names["vsmul.vv"],
        RISCVInstruction.classes_by_names["vsmul.vx"],


        # Not exact, but close enough
        RISCVInstruction.classes_by_names["vnmsub.vv"],
        RISCVInstruction.classes_by_names["vnmsub.vx"],
        RISCVInstruction.classes_by_names["vnmsac.vv"],
        RISCVInstruction.classes_by_names["vnmsac.vx"],

    ): lambda obj: obj.lmul_external * 1,

    (
        RISCVInstruction.classes_by_names["vand.vv"],
        RISCVInstruction.classes_by_names["vand.vx"],
        RISCVInstruction.classes_by_names["vand.vi"],
        RISCVInstruction.classes_by_names["vor.vv"],
        RISCVInstruction.classes_by_names["vor.vx"],
        RISCVInstruction.classes_by_names["vor.vi"],
        RISCVInstruction.classes_by_names["vxor.vv"],
        RISCVInstruction.classes_by_names["vxor.vx"],
        RISCVInstruction.classes_by_names["vxor.vi"],
        RISCVInstruction.classes_by_names["vnot.v"],
        RISCVInstruction.classes_by_names["vrgather.vx"],
        RISCVInstruction.classes_by_names["vrgather.vi"],
        RISCVInstruction.classes_by_names["vslideup.vx"],
        RISCVInstruction.classes_by_names["vslideup.vi"],
        RISCVInstruction.classes_by_names["vslide1up.vx"],

        # Simplified
        RISCVInstruction.classes_by_names["vadc.vvm"],
        RISCVInstruction.classes_by_names["vadc.vxm"],
        RISCVInstruction.classes_by_names["vadc.vim"],
        RISCVInstruction.classes_by_names["vsbc.vvm"],
        RISCVInstruction.classes_by_names["vsbc.vxm"],

        RISCVInstruction.classes_by_names["vmerge.vvm"],
        RISCVInstruction.classes_by_names["vmerge.vxm"],
        RISCVInstruction.classes_by_names["vmerge.vim"],

        RISCVInstruction.classes_by_names["vmv.v.v"],
        RISCVInstruction.classes_by_names["vmv.v.x"],
        RISCVInstruction.classes_by_names["vmv.v.i"],

        RISCVInstruction.classes_by_names["vsll.vv"],
        RISCVInstruction.classes_by_names["vsll.vx"],
        RISCVInstruction.classes_by_names["vsll.vi"],

        RISCVInstruction.classes_by_names["vsrl.vv"],
        RISCVInstruction.classes_by_names["vsrl.vx"],
        RISCVInstruction.classes_by_names["vsrl.vi"],
        RISCVInstruction.classes_by_names["vsra.vv"],
        RISCVInstruction.classes_by_names["vsra.vx"],
        RISCVInstruction.classes_by_names["vsra.vi"],
        RISCVInstruction.classes_by_names["vssrl.vv"],
        RISCVInstruction.classes_by_names["vssrl.vx"],
        RISCVInstruction.classes_by_names["vssrl.vi"],

        RISCVInstruction.classes_by_names["vwaddu.vv"],
        RISCVInstruction.classes_by_names["vwaddu.vx"],
        RISCVInstruction.classes_by_names["vwadd.vv"],
        RISCVInstruction.classes_by_names["vwadd.vx"],
        RISCVInstruction.classes_by_names["vwsub.vv"],
        RISCVInstruction.classes_by_names["vwsub.vx"],
        RISCVInstruction.classes_by_names["vwaddu.wv"],
        RISCVInstruction.classes_by_names["vwaddu.wx"],
        RISCVInstruction.classes_by_names["vwadd.wv"],
        RISCVInstruction.classes_by_names["vwadd.wx"],
        RISCVInstruction.classes_by_names["vwsub.wv"],
        RISCVInstruction.classes_by_names["vwsub.wx"],
        RISCVInstruction.classes_by_names["vwmulu.vv"],
        RISCVInstruction.classes_by_names["vwmulu.vx"],
        RISCVInstruction.classes_by_names["vwmulsu.vv"],
        RISCVInstruction.classes_by_names["vwmul.vv"],
        RISCVInstruction.classes_by_names["vwmul.vx"],
        RISCVInstruction.classes_by_names["vwmaccu.vv"],
        RISCVInstruction.classes_by_names["vwmaccu.vx"],
        RISCVInstruction.classes_by_names["vwmacc.vv"],
        RISCVInstruction.classes_by_names["vwmacc.vx"],
        RISCVInstruction.classes_by_names["vwmaccsu.vv"],
        RISCVInstruction.classes_by_names["vwmaccsu.vx"],
        RISCVInstruction.classes_by_names["vwmaccus.vx"],

        RISCVInstruction.classes_by_names["vzext.vf"],
        RISCVInstruction.classes_by_names["vsext.vf"],

        RISCVInstruction.classes_by_names["viota.m"],
        RISCVInstruction.classes_by_names["vid.v"],

        RISCVInstruction.classes_by_names["vle.v"],
        RISCVInstruction.classes_by_names["vlse.v"],
        RISCVInstruction.classes_by_names["vl.v"],
        RISCVInstruction.classes_by_names["vlr.v"],
        RISCVInstruction.classes_by_names["vluxei.v"],
        RISCVInstruction.classes_by_names["vloxei.v"],
        RISCVInstruction.classes_by_names["vlseg.v"],
        RISCVInstruction.classes_by_names["vlsseg.v"],
        RISCVInstruction.classes_by_names["vluxseg.v"],
        RISCVInstruction.classes_by_names["vloxseg.v"],
        RISCVInstruction.classes_by_names["vse.v"],
        RISCVInstruction.classes_by_names["vsse.v"],
        RISCVInstruction.classes_by_names["vssseg.v"],
        RISCVInstruction.classes_by_names["vsseg.v"],
        RISCVInstruction.classes_by_names["vsuxseg.v"],
        RISCVInstruction.classes_by_names["vsoxseg.v"],
        RISCVInstruction.classes_by_names["vsuxei.v"],
        RISCVInstruction.classes_by_names["vsoxei.v"],
        RISCVInstruction.classes_by_names["vs.v"],
        RISCVInstruction.classes_by_names["vsr.v"], # TODO: Fix fairly inaccurate guesses

    ): lambda obj: obj.lmul_external * 2,

    # Widening instructions
    (
        RISCVInstruction.classes_by_names["vnsrl.wv"],
        RISCVInstruction.classes_by_names["vnsrl.wx"],
        RISCVInstruction.classes_by_names["vnsrl.wi"],
        RISCVInstruction.classes_by_names["vnsra.wv"],
        RISCVInstruction.classes_by_names["vnsra.wx"],
        RISCVInstruction.classes_by_names["vnsra.wi"],
        RISCVInstruction.classes_by_names["vnclipu.wv"],
        RISCVInstruction.classes_by_names["vnclipu.wx"],
        RISCVInstruction.classes_by_names["vnclipu.wi"],
    ): lambda obj: obj.lmul_external * 4,

    (
        RISCVInstruction.classes_by_names["vrgather.vv"]
    ): vrgather_vv_inverse_throughput,

    (
        RISCVInstruction.classes_by_names["vslidedown.vx"],
        RISCVInstruction.classes_by_names["vslidedown.vi"],
        RISCVInstruction.classes_by_names["vslide1down.vx"],
        RISCVInstruction.classes_by_names["vslide1down.vi"],
    ): vslidedown_inverse_throughput,

    (
        RISCVInstruction.classes_by_names["vmadc.vvm"],
        RISCVInstruction.classes_by_names["vmadc.vxm"],
        RISCVInstruction.classes_by_names["vmadc.vim"],
        RISCVInstruction.classes_by_names["vmsbc.vvm"],
        RISCVInstruction.classes_by_names["vmsbc.vxm"],

        # vmseq also follows this pattern
        RISCVInstruction.classes_by_names["vmseq.vv"],
        RISCVInstruction.classes_by_names["vmseq.vx"],
        RISCVInstruction.classes_by_names["vmseq.vi"],

        RISCVInstruction.classes_by_names["vmsne.vv"],
        RISCVInstruction.classes_by_names["vmsne.vx"],
        RISCVInstruction.classes_by_names["vmsne.vi"],

        RISCVInstruction.classes_by_names["vmsltu.vv"],
        RISCVInstruction.classes_by_names["vmsltu.vx"],
        RISCVInstruction.classes_by_names["vmslt.vv"],
        RISCVInstruction.classes_by_names["vmslt.vx"],
        RISCVInstruction.classes_by_names["vmsleu.vv"],
        RISCVInstruction.classes_by_names["vmsleu.vx"],
        RISCVInstruction.classes_by_names["vmsleu.vi"],
        RISCVInstruction.classes_by_names["vmsle.vv"],
        RISCVInstruction.classes_by_names["vmsle.vx"],
        RISCVInstruction.classes_by_names["vmsle.vi"],
        RISCVInstruction.classes_by_names["vmsgtu.vx"],
        RISCVInstruction.classes_by_names["vmsgtu.vi"],
        RISCVInstruction.classes_by_names["vmsgt.vx"],
        RISCVInstruction.classes_by_names["vmsgt.vi"],
    ): vmasbc_inverse_throughput,

    (
        RISCVInstruction.classes_by_names["vmulhu.vv"],
        RISCVInstruction.classes_by_names["vmulhu.vx"],
        RISCVInstruction.classes_by_names["vmul.vv"],
        RISCVInstruction.classes_by_names["vmul.vx"],
        RISCVInstruction.classes_by_names["vmulhsu.vv"],
        RISCVInstruction.classes_by_names["vmulhsu.vx"],
        RISCVInstruction.classes_by_names["vmulh.vv"],
        RISCVInstruction.classes_by_names["vmulh.vx"],
        RISCVInstruction.classes_by_names["vmadd.vv"],
        RISCVInstruction.classes_by_names["vmadd.vx"],
        RISCVInstruction.classes_by_names["vmacc.vv"],
        RISCVInstruction.classes_by_names["vmacc.vx"],
    ): lmul1_slow_sew64_inverse_throughput,

    (
        RISCVInstruction.classes_by_names["vmsbf.m"],
        RISCVInstruction.classes_by_names["vmsof.m"],
        RISCVInstruction.classes_by_names["vmsif.m"],
    ): lambda obj: max(1, obj.lmul_external / 2),

    (RISCVInstruction.classes_by_names["vcompress.vm"]): vcompress_inverse_throughput,
    (RISCVInstruction.classes_by_names["vdivu.vv"]): vdivuvv_inverse_throughput,
    (RISCVInstruction.classes_by_names["vdivu.vx"]): vdivuvx_inverse_throughput,
    (RISCVInstruction.classes_by_names["vdiv.vv"]): vdivvv_inverse_throughput,
    (RISCVInstruction.classes_by_names["vdiv.vx"]): vdivvx_inverse_throughput,
    (RISCVInstruction.classes_by_names["vremu.vv"]): vremuvv_inverse_throughput,
    (RISCVInstruction.classes_by_names["vremu.vx"]): vremuvx_inverse_throughput,
    (RISCVInstruction.classes_by_names["vrem.vv"]): vremvv_inverse_throughput,
    (RISCVInstruction.classes_by_names["vrem.vx"]): vremvx_inverse_throughput,

}


rv32_inverse_throughput = {
    RISCVInstruction.classes_by_names["addi"]: 1,
    RISCVInstruction.classes_by_names["srli"]: 1,
    RISCVInstruction.classes_by_names["slli"]: 1,
    RISCVInstruction.classes_by_names["srli"]: 1,
    RISCVInstruction.classes_by_names["srai"]: 1,
    RISCVInstruction.classes_by_names["add"]: 1,
    RISCVInstruction.classes_by_names["sll"]: 1,
    RISCVInstruction.classes_by_names["srl"]: 1,
    RISCVInstruction.classes_by_names["sub"]: 1,
    RISCVInstruction.classes_by_names["neg"]: 1,
    RISCVInstruction.classes_by_names["sra"]: 1,
    RISCVInstruction.classes_by_names["mul"]: 1,
    RISCVInstruction.classes_by_names["div"]: 2,
    RISCVInstruction.classes_by_names["divu"]: 2,
    RISCVInstruction.classes_by_names["rem"]: 2,
    RISCVInstruction.classes_by_names["remu"]: 2,
    RISCVInstruction.classes_by_names["rol"]: 1,  # TODO: estimated
    RISCVInstruction.classes_by_names["ror"]: 1,  # TODO: estimated
    RISCVInstruction.classes_by_names["rori"]: 1,  # TODO: estimated
    RISCVInstruction.classes_by_names["pack"]: 1,  # TODO: estimated
}

default_latencies = {
    RISCVIntegerRegisterRegister: 1,
    RISCVIntegerRegisterImmediate: 1,
    RISCVIntegerRegister: 1,  # For Zbkb and pseudo-instructions
    RISCVUType: 1,
    RISCVLoad: 3,
    RISCVStore: 1,
    RISCVIntegerRegisterRegisterMul: 4,  # not correct for div, rem
    RISCVLiPseudo: 1,  # Pseudo-instruction
    RISCVULaPseudo: 1,  # Pseudo-instruction
}

rv32_latencies = {
    RISCVInstruction.classes_by_names["addi"]: 1,
    RISCVInstruction.classes_by_names["srli"]: 1,
    RISCVInstruction.classes_by_names["slli"]: 1,
    RISCVInstruction.classes_by_names["srai"]: 1,
    RISCVInstruction.classes_by_names["add"]: 1,
    RISCVInstruction.classes_by_names["sll"]: 1,
    RISCVInstruction.classes_by_names["srl"]: 1,
    RISCVInstruction.classes_by_names["sub"]: 1,
    RISCVInstruction.classes_by_names["sra"]: 1,
    RISCVInstruction.classes_by_names["mul"]: 3,
    RISCVInstruction.classes_by_names["div"]: 4,
    RISCVInstruction.classes_by_names["divu"]: 4,
    RISCVInstruction.classes_by_names["rem"]: 4,
    RISCVInstruction.classes_by_names["remu"]: 4,
    RISCVInstruction.classes_by_names["rol"]: 1,  # TODO: estimated
    RISCVInstruction.classes_by_names["ror"]: 1,  # TODO: estimated
    RISCVInstruction.classes_by_names["rori"]: 1,  # TODO: estimated
    RISCVInstruction.classes_by_names["pack"]: 1,  # TODO: estimated
}


def get_latency(src, out_idx, dst):
    _ = out_idx  # out_idx unused
    _ = dst  # dst is unused

    multiplier = 1
    if isinstance(src, RISCVVectorInstruction):
        eu = lookup_multidict(execution_units, src)
        multiplier = 2 if isinstance(eu, list) and ExecutionUnit.VEC1 in eu else 1
        latency = lookup_multidict(inverse_throughput, src)
    elif src.is_32_bit():
        latency = lookup_multidict(rv32_latencies, src)
    else:
        latency = lookup_multidict(default_latencies, src)

    if isinstance(latency, int):
        return latency * multiplier
    else:
        return latency(src) * multiplier


def get_units(src):
    units = lookup_multidict(execution_units, src)
    if isinstance(units, list):
        return units
    return [units]


def get_inverse_throughput(src):
    if src.is_32_bit():
        throughput = lookup_multidict(rv32_inverse_throughput, src)
    else:
        throughput = lookup_multidict(inverse_throughput, src)

    if isinstance(throughput, int):
        return throughput
    else:
        return throughput(src)
