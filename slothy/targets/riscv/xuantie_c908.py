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
import os
################################### NOTE ###############################################  # noqa: E266
###                                                                                  ###  # noqa: E266
### WARNING: The data in this module is approximate and may contain errors.          ###  # noqa: E266
###          They are _NOT_ an official software optimization guide for C908.        ###  # noqa: E266
###                                                                                  ###  # noqa: E266
########################################################################################  # noqa: E266

from enum import Enum
from unittest import case
import csv

from slothy.targets.riscv.riscv import *  # noqa: F403
from slothy.targets.riscv.rv32_64_i_instructions import *  # noqa: F403
from slothy.targets.riscv.rv32_64_m_instructions import *  # noqa: F403
from slothy.targets.riscv.rv32_64_b_instructions import *  # noqa: F403
from slothy.targets.riscv.rv32_64_pseudo_instructions import *  # noqa: F403
from slothy.targets.riscv.rv32_64_v_instructions import RISCVVectorInstruction

issue_rate = 2
llvm_mca_target = ""

FOLDER = os.path.dirname(__file__)
xuantie_c908_vector_data = {}
with open(os.path.join(FOLDER, "data/xuantie_c908_vlmax_ta_ma.csv"), newline='') as csvfile:
    xuantie_c908_vector_csv = csv.reader(csvfile, delimiter=',')

    for line in xuantie_c908_vector_csv:
        xuantie_c908_vector_data[line[0]] = line[1:]


class ExecutionUnit(Enum):
    """Enumeration of execution units in C908 model"""

    SCALAR_ALU0 = 1
    SCALAR_ALU1 = 2
    SCALAR_MUL = 3
    LSU = 4
    VEC0 = 5
    VEC1 = 6
    VEC_LSU = 7

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

        RISCVInstruction.classes_by_names["vmv.s.x"],
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
        RISCVInstruction.classes_by_names["vmv.x.s"],

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
        RISCVInstruction.classes_by_names["vssra.vv"],
        RISCVInstruction.classes_by_names["vssra.vx"],
        RISCVInstruction.classes_by_names["vssra.vi"],

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

        RISCVInstruction.classes_by_names["vsetvl"],
        RISCVInstruction.classes_by_names["vsetvli"],
        RISCVInstruction.classes_by_names["vsetivli"]
    ): ExecutionUnit.VEC0,
    (
        # Vector load/store handled by single load/store unit
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
    ): ExecutionUnit.VEC_LSU,
}


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

    (
        RISCVInstruction.classes_by_names["vsetvl"],
        RISCVInstruction.classes_by_names["vsetvli"],
        RISCVInstruction.classes_by_names["vsetivli"]
    ): 4,
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
        multiplier = 2 if ExecutionUnit.VEC1 not in get_units(src) else 1
        latency = get_inverse_throughput(src)
    elif src.is_32_bit():
        latency = lookup_multidict(rv32_latencies, src)
    else:
        latency = lookup_multidict(default_latencies, src)

    if isinstance(latency, int):
        return latency * multiplier
    else:
        return latency(src) * multiplier


def get_latency(src, out_idx, dst):
    _ = out_idx  # out_idx unused
    _ = dst  # dst is unused

    if isinstance(src, RISCVVectorInstruction):
        return max(4, round(get_inverse_throughput_exact(src) * len(get_units(src))))
    elif src.is_32_bit():
        return lookup_multidict(rv32_latencies, src)
    else:
        return lookup_multidict(default_latencies, src)

def get_units(src):
    units = lookup_multidict(execution_units, src)
    if isinstance(units, list):
        return units
    return [units]

def get_inverse_throughput_exact(src):
    if isinstance(src, RISCVVectorInstruction):
        instruction = src.pattern.split(" ")[0]
        instruction = instruction.replace("<len>", str(getattr(src, "len", 32)))
        instruction = instruction.replace("<nf>", str(getattr(src, "nf", 1)))

        if instruction in xuantie_c908_vector_data:
            sew_values = [8, 16, 32, 64]
            lmul_values = [0.125, 0.25, 0.5, 1, 2, 4, 8]
            # Just fix the lmul and sew for now, to avoid instructions with really high runtime
            throughput = xuantie_c908_vector_data[instruction][7 * sew_values.index(32) + lmul_values.index(1)]
            return float(throughput)

    return lookup_multidict(inverse_throughput, src)

def get_inverse_throughput(src):
    return round(get_inverse_throughput_exact(src))