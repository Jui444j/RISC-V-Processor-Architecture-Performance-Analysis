import os
import argparse
import copy
import sys
from typing import Tuple, List


MemSize = 1000  # bytes
NOP_INSTR = "00000000000000000000000000010011"  # ADDI x0, x0, 0
LW_FUNCT3S = {"000", "010"}  # support project sheet + uploaded testcase


# ------------------------------- bit helpers ---------------------------------

def mask_u32(x: int) -> int:
    return x & 0xFFFFFFFF


def u32_to_bin(x: int) -> str:
    return format(mask_u32(x), "032b")


def sign_extend(value: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    value &= (1 << bits) - 1
    return (value ^ sign_bit) - sign_bit


def bits_to_u32(bits: str) -> int:
    return int(bits, 2) & 0xFFFFFFFF


def decode_common(instr: str):
    opcode = int(instr[25:32], 2)
    rd = int(instr[20:25], 2)
    funct3 = instr[17:20]
    rs1 = int(instr[12:17], 2)
    rs2 = int(instr[7:12], 2)
    funct7 = instr[0:7]
    return opcode, rd, rs1, rs2, funct3, funct7


def imm_i(instr: str) -> int:
    return sign_extend(int(instr[0:12], 2), 12)


def imm_s(instr: str) -> int:
    return sign_extend(int(instr[0:7] + instr[20:25], 2), 12)


def imm_b(instr: str) -> int:
    imm_str = instr[0] + instr[24] + instr[1:7] + instr[20:24] + "0"
    return sign_extend(int(imm_str, 2), 13)


def imm_j(instr: str) -> int:
    imm_str = instr[0] + instr[12:20] + instr[11] + instr[1:11] + "0"
    return sign_extend(int(imm_str, 2), 21)


# ---------------------------- memory + RF models -----------------------------

class InsMem(object):
    def __init__(self, name: str, ioDir: str):
        self.id = name
        with open(os.path.join(ioDir, "imem.txt")) as im:
            self.IMem = [data.strip() for data in im.readlines()]

    def readInstr(self, ReadAddress: int) -> str:
        if ReadAddress < 0:
            return NOP_INSTR
        if ReadAddress <= len(self.IMem) - 4:
            return "".join(self.IMem[ReadAddress:ReadAddress + 4])
        return NOP_INSTR


class DataMem(object):
    def __init__(self, name: str, ioDir: str):
        self.id = name
        self.ioDir = ioDir
        with open(os.path.join(ioDir, "dmem.txt")) as dm:
            self.DMem = [data.strip() for data in dm.readlines()]
            for _ in range(MemSize - len(self.DMem)):
                self.DMem.append("00000000")

    def _check_word_addr(self, addr: int):
        if addr < 0 or addr + 3 >= MemSize:
            raise IndexError(f"DataMem access out of range: addr={addr}")

    def readDataMem(self, ReadAddress: int) -> int:
        self._check_word_addr(ReadAddress)
        return int("".join(self.DMem[ReadAddress:ReadAddress + 4]), 2) & 0xFFFFFFFF

    def writeDataMem(self, Address: int, WriteData: int):
        self._check_word_addr(Address)
        data = u32_to_bin(WriteData)
        for i in range(4):
            self.DMem[Address + i] = data[i * 8:(i + 1) * 8]

    def outputDataMem(self):
        resPath = os.path.join(self.ioDir, f"{self.id}_DMEMResult.txt")
        with open(resPath, "w") as rp:
            rp.writelines([str(data) + "\n" for data in self.DMem])


class RegisterFile(object):
    def __init__(self, ioPrefix: str):
        self.outputFile = ioPrefix + "RFResult.txt"
        self.Registers = ["0" * 32 for _ in range(32)]

    def readRF(self, Reg_addr: int) -> int:
        return int(self.Registers[Reg_addr], 2) & 0xFFFFFFFF

    def writeRF(self, Reg_addr: int, Wrt_reg_data: int):
        if Reg_addr > 0:
            self.Registers[Reg_addr] = u32_to_bin(Wrt_reg_data)

    def outputRF(self, cycle: int):
        op = ["-" * 70 + "\n", "State of RF after executing cycle:" + str(cycle) + "\n"]
        op.extend([str(val) + "\n" for val in self.Registers])
        perm = "w" if cycle == 0 else "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)


# ------------------------------- core states ---------------------------------

class State(object):
    def __init__(self):
        self.IF = {"nop": False, "PC": 0}
        self.ID = {"nop": True, "Instr": NOP_INSTR}
        self.EX = {
            "nop": True,
            "instr": "",
            "Read_data1": "0" * 32,
            "Read_data2": "0" * 32,
            "Imm": "0" * 32,
            "Rs": "0" * 5,
            "Rt": "0" * 5,
            "Wrt_reg_addr": "0" * 5,
            "is_I_type": 0,
            "rd_mem": 0,
            "wrt_mem": 0,
            "alu_op": "00",
            "wrt_enable": 0
        }
        self.MEM = {
            "nop": True,
            "ALUresult": "0" * 32,
            "Store_data": "0" * 32,
            "Rs": "0" * 5,
            "Rt": "0" * 5,
            "Wrt_reg_addr": "0" * 5,
            "rd_mem": 0,
            "wrt_mem": 0,
            "wrt_enable": 0
        }
        self.WB = {
            "nop": True,
            "Wrt_data": "0" * 32,
            "Rs": "0" * 5,
            "Rt": "0" * 5,
            "Wrt_reg_addr": "0" * 5,
            "wrt_enable": 0
        }


class Core(object):
    def __init__(self, ioPrefix: str, imem: InsMem, dmem: DataMem):
        self.myRF = RegisterFile(ioPrefix)
        self.cycle = 0
        self.halted = False
        self.ioPrefix = ioPrefix
        self.state = State()
        self.nextState = State()
        self.ext_imem = imem
        self.ext_dmem = dmem
        self.instr_count = 0


# ---------------------------- single-stage model -----------------------------

class SingleStageCore(Core):
    def __init__(self, ioDir: str, imem: InsMem, dmem: DataMem):
        super(SingleStageCore, self).__init__(os.path.join(ioDir, "SS_"), imem, dmem)
        self.opFilePath = os.path.join(ioDir, "StateResult_SS.txt")

    def step(self):
        if self.halted:
            return

        # one extra final visible cycle after IF.nop becomes True
        if self.state.IF["nop"]:
            self.nextState = copy.deepcopy(self.state)
            self.myRF.outputRF(self.cycle)
            self.printState(self.nextState, self.cycle)
            self.state = copy.deepcopy(self.nextState)
            self.cycle += 1
            self.halted = True
            return

        self.nextState = copy.deepcopy(self.state)

        pc = self.state.IF["PC"]
        instr = self.ext_imem.readInstr(pc)
        opcode, rd, rs1, rs2, funct3, funct7 = decode_common(instr)

        next_pc = pc + 4

        # HALT
        if opcode == 0b1111111:
            self.instr_count += 1
            self.nextState.IF["PC"] = pc
            self.nextState.IF["nop"] = True

        # R-type
        elif opcode == 0b0110011:
            a = self.myRF.readRF(rs1)
            b = self.myRF.readRF(rs2)

            if funct3 == "000" and funct7 == "0000000":
                self.myRF.writeRF(rd, a + b)   # ADD
            elif funct3 == "000" and funct7 == "0100000":
                self.myRF.writeRF(rd, a - b)   # SUB
            elif funct3 == "100" and funct7 == "0000000":
                self.myRF.writeRF(rd, a ^ b)   # XOR
            elif funct3 == "110" and funct7 == "0000000":
                self.myRF.writeRF(rd, a | b)   # OR
            elif funct3 == "111" and funct7 == "0000000":
                self.myRF.writeRF(rd, a & b)   # AND
            else:
                raise ValueError(f"Unsupported R-type encoding: funct3={funct3}, funct7={funct7}")

            self.instr_count += 1
            self.nextState.IF["PC"] = next_pc
            self.nextState.IF["nop"] = False

        # I-type arithmetic
        elif opcode == 0b0010011:
            a = self.myRF.readRF(rs1)
            imm = imm_i(instr)
            imm_u = mask_u32(imm)

            if funct3 == "000":
                self.myRF.writeRF(rd, a + imm)       # ADDI
            elif funct3 == "100":
                self.myRF.writeRF(rd, a ^ imm_u)     # XORI
            elif funct3 == "110":
                self.myRF.writeRF(rd, a | imm_u)     # ORI
            elif funct3 == "111":
                self.myRF.writeRF(rd, a & imm_u)     # ANDI
            else:
                raise ValueError(f"Unsupported I-type funct3={funct3}")

            self.instr_count += 1
            self.nextState.IF["PC"] = next_pc
            self.nextState.IF["nop"] = False

        # LW
        elif opcode == 0b0000011 and funct3 in LW_FUNCT3S:
            base = self.myRF.readRF(rs1)
            offset = imm_i(instr)
            addr = mask_u32(base + offset)
            data = self.ext_dmem.readDataMem(addr)
            self.myRF.writeRF(rd, data)
            self.instr_count += 1
            self.nextState.IF["PC"] = next_pc
            self.nextState.IF["nop"] = False

        # SW
        elif opcode == 0b0100011 and funct3 == "010":
            base = self.myRF.readRF(rs1)
            offset = imm_s(instr)
            addr = mask_u32(base + offset)
            data = self.myRF.readRF(rs2)
            self.ext_dmem.writeDataMem(addr, data)
            self.instr_count += 1
            self.nextState.IF["PC"] = next_pc
            self.nextState.IF["nop"] = False

        # BEQ / BNE
        elif opcode == 0b1100011:
            a = self.myRF.readRF(rs1)
            b = self.myRF.readRF(rs2)
            offset = imm_b(instr)

            if funct3 == "000":
                if a == b:
                    next_pc = pc + offset
            elif funct3 == "001":
                if a != b:
                    next_pc = pc + offset
            else:
                raise ValueError(f"Unsupported branch funct3={funct3}")

            self.instr_count += 1
            self.nextState.IF["PC"] = next_pc
            self.nextState.IF["nop"] = False

        # JAL
        elif opcode == 0b1101111:
            offset = imm_j(instr)
            self.myRF.writeRF(rd, pc + 4)
            next_pc = pc + offset
            self.instr_count += 1
            self.nextState.IF["PC"] = next_pc
            self.nextState.IF["nop"] = False

        else:
            raise ValueError(
                f"Unsupported instruction in single-stage core: opcode={opcode:07b}, funct3={funct3}"
            )

        self.myRF.outputRF(self.cycle)
        self.printState(self.nextState, self.cycle)

        self.state = copy.deepcopy(self.nextState)
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["-" * 70 + "\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.append("IF.PC: " + str(state.IF["PC"]) + "\n")
        printstate.append("IF.nop: " + str(state.IF["nop"]) + "\n")

        perm = "w" if cycle == 0 else "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)


# ---------------------------- five-stage pipeline ----------------------------

class FiveStageCore(Core):
    def __init__(self, ioDir: str, imem: InsMem, dmem: DataMem):
        super(FiveStageCore, self).__init__(os.path.join(ioDir, "FS_"), imem, dmem)
        self.opFilePath = os.path.join(ioDir, "StateResult_FS.txt")
        self.fetch_stopped = False

    @staticmethod
    def _reg5(x: int) -> str:
        return format(x & 0x1F, "05b")

    def _sources_for_hazard(self, instr: str) -> List[int]:
        opcode, rd, rs1, rs2, funct3, funct7 = decode_common(instr)

        if opcode == 0b0110011:
            return [rs1, rs2]
        if opcode == 0b0010011:
            return [rs1]
        if opcode == 0b0000011:
            return [rs1]
        if opcode == 0b0100011:
            return [rs1, rs2]
        if opcode == 0b1100011:
            return [rs1, rs2]
        if opcode == 0b1101111:
            return []
        if opcode == 0b1111111:
            return []
        return [rs1, rs2]

    def step(self):
        if self.halted:
            return

        # one extra final visible cycle after pipeline fully drains
        if (
            self.fetch_stopped
            and self.state.IF["nop"]
            and self.state.ID["nop"]
            and self.state.EX["nop"]
            and self.state.MEM["nop"]
            and self.state.WB["nop"]
        ):
            self.nextState = copy.deepcopy(self.state)
            self.myRF.outputRF(self.cycle)
            self.printState(self.nextState, self.cycle)
            self.state = copy.deepcopy(self.nextState)
            self.cycle += 1
            self.halted = True
            return

        # keep old fields unless overwritten; this matches the uploaded reference
        self.nextState = copy.deepcopy(self.state)

        mem_forward_valid = False
        mem_forward_reg = 0
        mem_forward_value = 0

        # --------------------- WB stage ---------------------
        if not self.state.WB["nop"]:
            self.instr_count += 1
            if self.state.WB["wrt_enable"]:
                rd = int(self.state.WB["Wrt_reg_addr"], 2)
                if rd != 0:
                    self.myRF.writeRF(rd, bits_to_u32(self.state.WB["Wrt_data"]))

        # --------------------- MEM stage --------------------
        if self.state.MEM["nop"]:
            self.nextState.WB["nop"] = True
        else:
            self.nextState.WB["nop"] = False
            self.nextState.WB["Wrt_reg_addr"] = self.state.MEM["Wrt_reg_addr"]
            self.nextState.WB["wrt_enable"] = self.state.MEM["wrt_enable"]

            addr = bits_to_u32(self.state.MEM["ALUresult"])

            if self.state.MEM["wrt_mem"]:
                self.ext_dmem.writeDataMem(addr, bits_to_u32(self.state.MEM["Store_data"]))

            if self.state.MEM["rd_mem"]:
                data = self.ext_dmem.readDataMem(addr)
                self.nextState.WB["Wrt_data"] = u32_to_bin(data)
                mem_forward_value = data
            else:
                self.nextState.WB["Wrt_data"] = self.state.MEM["ALUresult"]
                mem_forward_value = bits_to_u32(self.state.MEM["ALUresult"])

            mem_forward_valid = bool(self.state.MEM["wrt_enable"])
            mem_forward_reg = int(self.state.MEM["Wrt_reg_addr"], 2)

        # --------------------- EX stage ---------------------
        ex_result = 0
        ex_wrt_enable = 0
        ex_wrt_reg = 0

        if self.state.EX["nop"]:
            self.nextState.MEM["nop"] = True
        else:
            self.nextState.MEM["nop"] = False
            self.nextState.MEM["Wrt_reg_addr"] = self.state.EX["Wrt_reg_addr"]
            self.nextState.MEM["wrt_enable"] = self.state.EX["wrt_enable"]
            self.nextState.MEM["rd_mem"] = self.state.EX["rd_mem"]
            self.nextState.MEM["wrt_mem"] = self.state.EX["wrt_mem"]

            rs1 = int(self.state.EX["Rs"], 2)
            rs2 = int(self.state.EX["Rt"], 2)
            ex_wrt_enable = int(self.state.EX["wrt_enable"])
            ex_wrt_reg = int(self.state.EX["Wrt_reg_addr"], 2)

            a = bits_to_u32(self.state.EX["Read_data1"])
            b = bits_to_u32(self.state.EX["Read_data2"])

            # Forward from MEM stage only for non-load results
            if not self.state.MEM["nop"]:
                mem_rd = int(self.state.MEM["Wrt_reg_addr"], 2)
                if self.state.MEM["wrt_enable"] and mem_rd != 0 and not self.state.MEM["rd_mem"]:
                    mem_val = bits_to_u32(self.state.MEM["ALUresult"])
                    if mem_rd == rs1:
                        a = mem_val
                    if mem_rd == rs2:
                        b = mem_val

            # Forward from WB stage
            if not self.state.WB["nop"]:
                wb_rd = int(self.state.WB["Wrt_reg_addr"], 2)
                if self.state.WB["wrt_enable"] and wb_rd != 0:
                    wb_val = bits_to_u32(self.state.WB["Wrt_data"])
                    if wb_rd == rs1:
                        a = wb_val
                    if wb_rd == rs2:
                        b = wb_val

            store_data = b

            if self.state.EX["is_I_type"]:
                b = bits_to_u32(self.state.EX["Imm"])

            op = self.state.EX["alu_op"]
            if op == "00":
                ex_result = mask_u32(a + b)
            elif op == "01":
                ex_result = mask_u32(a - b)
            elif op == "10":
                ex_result = mask_u32(a & b)
            elif op == "11":
                ex_result = mask_u32(a | b)
            elif op == "xor":
                ex_result = mask_u32(a ^ b)
            else:
                ex_result = 0

            self.nextState.MEM["ALUresult"] = u32_to_bin(ex_result)
            self.nextState.MEM["Store_data"] = u32_to_bin(store_data)
            self.nextState.MEM["Rs"] = self.state.EX["Rs"]
            self.nextState.MEM["Rt"] = self.state.EX["Rt"]

        # --------------------- ID stage ---------------------
        stall = False
        control_taken = False
        control_target_pc = 0
        fetched_halt_this_cycle = False

        if self.state.ID["nop"]:
            self.nextState.EX["nop"] = True
        else:
            instr = self.state.ID["Instr"]
            opcode, rd, rs1, rs2, funct3, funct7 = decode_common(instr)

            # Load-use hazard
            if (not self.state.EX["nop"]) and self.state.EX["rd_mem"]:
                load_rd = int(self.state.EX["Wrt_reg_addr"], 2)
                if load_rd != 0 and load_rd in self._sources_for_hazard(instr):
                    stall = True

            if stall:
                self.nextState.EX["nop"] = True
                self.nextState.EX["instr"] = instr
                self.nextState.ID["nop"] = False
                self.nextState.ID["Instr"] = self.state.ID["Instr"]
            else:
                self.nextState.EX["nop"] = False
                self.nextState.EX["instr"] = instr
                self.nextState.EX["Rs"] = self._reg5(rs1)
                self.nextState.EX["Rt"] = self._reg5(rs2)
                self.nextState.EX["Wrt_reg_addr"] = self._reg5(rd)

                r1 = self.myRF.readRF(rs1)
                r2 = self.myRF.readRF(rs2)

                br_a = r1
                br_b = r2

                # EX-ID forwarding for branches
                if (not self.state.EX["nop"]) and ex_wrt_enable and ex_wrt_reg != 0 and not self.state.EX["rd_mem"]:
                    if ex_wrt_reg == rs1:
                        br_a = ex_result
                    if ex_wrt_reg == rs2:
                        br_b = ex_result

                # MEM-ID forwarding
                if (not self.state.MEM["nop"]) and mem_forward_valid and mem_forward_reg != 0:
                    if mem_forward_reg == rs1:
                        br_a = mem_forward_value
                    if mem_forward_reg == rs2:
                        br_b = mem_forward_value

                # WB-ID forwarding
                if not self.state.WB["nop"] and self.state.WB["wrt_enable"]:
                    wb_rd = int(self.state.WB["Wrt_reg_addr"], 2)
                    if wb_rd != 0:
                        wb_val = bits_to_u32(self.state.WB["Wrt_data"])
                        if wb_rd == rs1:
                            br_a = wb_val
                        if wb_rd == rs2:
                            br_b = wb_val

                self.nextState.EX["is_I_type"] = 0
                self.nextState.EX["rd_mem"] = 0
                self.nextState.EX["wrt_mem"] = 0
                self.nextState.EX["alu_op"] = "00"
                self.nextState.EX["wrt_enable"] = 0

                id_pc = self.state.IF["PC"] - 4

                # R-type
                if opcode == 0b0110011:
                    self.nextState.EX["Read_data1"] = u32_to_bin(r1)
                    self.nextState.EX["Read_data2"] = u32_to_bin(r2)
                    self.nextState.EX["wrt_enable"] = 1

                    if funct3 == "000" and funct7 == "0000000":
                        self.nextState.EX["alu_op"] = "00"
                    elif funct3 == "000" and funct7 == "0100000":
                        self.nextState.EX["alu_op"] = "01"
                    elif funct3 == "100" and funct7 == "0000000":
                        self.nextState.EX["alu_op"] = "xor"
                    elif funct3 == "110" and funct7 == "0000000":
                        self.nextState.EX["alu_op"] = "11"
                    elif funct3 == "111" and funct7 == "0000000":
                        self.nextState.EX["alu_op"] = "10"
                    else:
                        sys.exit("Error: Unsupported R-type instruction")

                # I-type arithmetic
                elif opcode == 0b0010011:
                    imm = imm_i(instr)
                    self.nextState.EX["Read_data1"] = u32_to_bin(r1)
                    self.nextState.EX["Read_data2"] = u32_to_bin(r2)
                    self.nextState.EX["Imm"] = u32_to_bin(imm)
                    self.nextState.EX["is_I_type"] = 1
                    self.nextState.EX["wrt_enable"] = 1

                    if funct3 == "000":
                        self.nextState.EX["alu_op"] = "00"
                    elif funct3 == "100":
                        self.nextState.EX["alu_op"] = "xor"
                    elif funct3 == "110":
                        self.nextState.EX["alu_op"] = "11"
                    elif funct3 == "111":
                        self.nextState.EX["alu_op"] = "10"
                    else:
                        sys.exit("Error: Unsupported I-type arithmetic")

                # LW
                elif opcode == 0b0000011 and funct3 in LW_FUNCT3S:
                    imm = imm_i(instr)
                    self.nextState.EX["Read_data1"] = u32_to_bin(r1)
                    self.nextState.EX["Read_data2"] = u32_to_bin(r2)
                    self.nextState.EX["Imm"] = u32_to_bin(imm)
                    self.nextState.EX["is_I_type"] = 1
                    self.nextState.EX["rd_mem"] = 1
                    self.nextState.EX["alu_op"] = "00"
                    self.nextState.EX["wrt_enable"] = 1

                # SW
                elif opcode == 0b0100011 and funct3 == "010":
                    imm = imm_s(instr)
                    self.nextState.EX["Read_data1"] = u32_to_bin(r1)
                    self.nextState.EX["Read_data2"] = u32_to_bin(r2)
                    self.nextState.EX["Imm"] = u32_to_bin(imm)
                    self.nextState.EX["is_I_type"] = 1
                    self.nextState.EX["wrt_mem"] = 1
                    self.nextState.EX["alu_op"] = "00"
                    self.nextState.EX["wrt_enable"] = 0
                    self.nextState.EX["Wrt_reg_addr"] = "00000"

                # BEQ / BNE
                elif opcode == 0b1100011:
                    self.instr_count += 1
                    off = imm_b(instr)
                    taken = False

                    if funct3 == "000":
                        taken = (br_a == br_b)
                    elif funct3 == "001":
                        taken = (br_a != br_b)
                    else:
                        sys.exit("Error: Unsupported BRANCH funct3")

                    # Branches are resolved in ID/RF, so they do not consume EX.
                    # Keep the instruction/operand fields visible in the state dump,
                    # but mark the EX stage as a bubble for the next cycle.
                    self.nextState.EX["nop"] = True
                    self.nextState.EX["wrt_enable"] = 0
                    self.nextState.EX["Wrt_reg_addr"] = "00000"
                    self.nextState.EX["Read_data1"] = u32_to_bin(br_a)
                    self.nextState.EX["Read_data2"] = u32_to_bin(br_b)
                    self.nextState.EX["Imm"] = format(off & 0x1FFF, "013b")

                    if taken:
                        control_taken = True
                        control_target_pc = id_pc + off

                # JAL
                elif opcode == 0b1101111:
                    off = imm_j(instr)
                    control_taken = True
                    control_target_pc = id_pc + off

                    self.nextState.EX["Read_data1"] = "0" * 32
                    self.nextState.EX["Read_data2"] = "0" * 32
                    self.nextState.EX["Imm"] = u32_to_bin(id_pc + 4)
                    self.nextState.EX["is_I_type"] = 1
                    self.nextState.EX["alu_op"] = "00"
                    self.nextState.EX["wrt_enable"] = 1

                # HALT should not normally appear here because we stop it in IF
                elif opcode == 0b1111111:
                    self.nextState.EX["nop"] = True
                    self.nextState.ID["nop"] = True
                    self.fetch_stopped = True

                else:
                    sys.exit(
                        f"Error: Unsupported instruction in five-stage core: opcode={opcode:07b}, funct3={funct3}"
                    )

        # --------------------- IF stage ---------------------
        if self.fetch_stopped:
            self.nextState.IF["nop"] = True
            self.nextState.IF["PC"] = self.state.IF["PC"]
            self.nextState.ID["nop"] = True
            self.nextState.ID["Instr"] = self.state.ID["Instr"]

        elif self.state.IF["nop"]:
            self.nextState.IF["nop"] = True
            self.nextState.IF["PC"] = self.state.IF["PC"]
            self.nextState.ID["nop"] = True
            self.nextState.ID["Instr"] = self.state.ID["Instr"]

        elif stall:
            self.nextState.IF["nop"] = False
            self.nextState.IF["PC"] = self.state.IF["PC"]
            self.nextState.ID["nop"] = False
            self.nextState.ID["Instr"] = self.state.ID["Instr"]

        else:
            fetched = self.ext_imem.readInstr(self.state.IF["PC"])

            # detect HALT in IF and stop fetch immediately, matching the uploaded reference
            if decode_common(fetched)[0] == 0b1111111:
                fetched_halt_this_cycle = True
                self.fetch_stopped = True
                self.instr_count += 1  # count HALT
                self.nextState.IF["nop"] = True
                self.nextState.IF["PC"] = self.state.IF["PC"]
                self.nextState.ID["nop"] = True
                self.nextState.ID["Instr"] = self.state.ID["Instr"]
            else:
                self.nextState.ID["nop"] = False
                self.nextState.ID["Instr"] = fetched
                self.nextState.IF["nop"] = False
                self.nextState.IF["PC"] = self.state.IF["PC"] + 4

        # Taken branch/JAL flush. A taken control instruction in ID must override a
        # wrong-path HALT fetched in IF during the same cycle.
        if control_taken and not stall:
            if fetched_halt_this_cycle:
                self.fetch_stopped = False
                self.instr_count -= 1
            self.nextState.ID["nop"] = True
            self.nextState.ID["Instr"] = self.state.ID["Instr"]
            self.nextState.IF["PC"] = control_target_pc
            self.nextState.IF["nop"] = False

        self.myRF.outputRF(self.cycle)
        self.printState(self.nextState, self.cycle)

        self.state = copy.deepcopy(self.nextState)
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["-" * 70 + "\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.extend(["IF." + key + ": " + str(val) + "\n" for key, val in state.IF.items()])
        printstate.extend(["ID." + key + ": " + str(val) + "\n" for key, val in state.ID.items()])
        printstate.extend(["EX." + key + ": " + str(val) + "\n" for key, val in state.EX.items()])
        printstate.extend(["MEM." + key + ": " + str(val) + "\n" for key, val in state.MEM.items()])
        printstate.extend(["WB." + key + ": " + str(val) + "\n" for key, val in state.WB.items()])

        perm = "w" if cycle == 0 else "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)


# ------------------------------- entry point ---------------------------------

def run_simulation(iodir: str) -> Tuple[SingleStageCore, FiveStageCore]:
    ioDir = os.path.abspath(iodir)
    print("IO Directory:", ioDir)

    imem = InsMem("Imem", ioDir)
    dmem_ss = DataMem("SS", ioDir)
    dmem_fs = DataMem("FS", ioDir)

    ssCore = SingleStageCore(ioDir, imem, dmem_ss)
    fsCore = FiveStageCore(ioDir, imem, dmem_fs)

    while True:
        if not ssCore.halted:
            ssCore.step()
        if not fsCore.halted:
            fsCore.step()
        if ssCore.halted and fsCore.halted:
            break

    print("\n--- Performance Metrics ---\n")

    print("Single Stage Core Metrics:")
    print("Total Execution Cycles:", ssCore.cycle)
    print("Total Instructions Executed:", ssCore.instr_count)
    if ssCore.instr_count > 0:
        print("CPI (Cycle Per Instruction):", ssCore.cycle / ssCore.instr_count)
        print("IPC (Instruction Per Cycle):", ssCore.instr_count / ssCore.cycle)
    else:
        print("No instructions executed.")

    print("\nFive Stage Core Metrics:")
    print("Total Execution Cycles:", fsCore.cycle)
    print("Total Instructions Executed:", fsCore.instr_count)
    if fsCore.instr_count > 0:
        print("CPI (Cycle Per Instruction):", fsCore.cycle / fsCore.instr_count)
        print("IPC (Instruction Per Cycle):", fsCore.instr_count / fsCore.cycle)
    else:
        print("No instructions executed.")

    dmem_ss.outputDataMem()
    dmem_fs.outputDataMem()

    return ssCore, fsCore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RV32I processor")
    parser.add_argument("--iodir", default="", type=str, help="Directory containing the input files.")
    args = parser.parse_args()
    run_simulation(args.iodir)