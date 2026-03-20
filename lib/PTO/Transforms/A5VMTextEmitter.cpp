//===- A5VMTextEmitter.cpp - A5VM textual LLVM-like emitter --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/A5VMTextEmitter.h"

#include "PTO/IR/A5VM.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;

namespace mlir::pto {
namespace {

static LogicalResult writeReportFile(const A5VMEmissionOptions &options,
                                     ArrayRef<std::string> unresolved) {
  if (options.unresolvedReportPath.empty())
    return success();

  std::error_code ec;
  llvm::raw_fd_ostream reportOS(options.unresolvedReportPath, ec,
                                llvm::sys::fs::OF_Text);
  if (ec)
    return failure();
  for (const std::string &record : unresolved)
    reportOS << record << "\n";
  return success();
}

static constexpr llvm::StringLiteral kSupportedA5VMPrimitives[] = {
    "a5vm.set_flag",              "a5vm.wait_flag",
    "a5vm.pipe_barrier",          "a5vm.get_buf",
    "a5vm.rls_buf",               "a5vm.set_loop2_stride_outtoub",
    "a5vm.set_loop1_stride_outtoub",
    "a5vm.set_loop_size_outtoub", "a5vm.set_loop2_stride_ubtoout",
    "a5vm.set_loop1_stride_ubtoout",
    "a5vm.set_loop_size_ubtoout", "a5vm.copy_gm_to_ubuf",
    "a5vm.copy_ubuf_to_ubuf",
    "a5vm.vlds",                  "a5vm.vldas",
    "a5vm.vldus",                 "a5vm.vbr",
    "a5vm.vdup",                  "a5vm.vci",
    "a5vm.vbitsort",              "a5vm.vmrgsort4",
    "a5vm.vgather2",              "a5vm.vgatherb",
    "a5vm.vgather2_bc",
    "a5vm.pset_b8",
    "a5vm.pset_b16",
    "a5vm.pset_b32",
    "a5vm.pge_b8",
    "a5vm.pge_b16",
    "a5vm.pge_b32",
    "a5vm.ppack",                 "a5vm.punpack",
    "a5vm.pnot",
    "a5vm.psel",
    "a5vm.plds",                  "a5vm.pld",
    "a5vm.pldi",                  "a5vm.pst",
    "a5vm.psti",
    "a5vm.vabs",                  "a5vm.vcadd",
    "a5vm.vcmax",                 "a5vm.vcmin",
    "a5vm.vexp",                  "a5vm.vln",
    "a5vm.vsqrt",                 "a5vm.vrec",
    "a5vm.vrelu",                 "a5vm.vnot",
    "a5vm.vadd",                  "a5vm.vsub",
    "a5vm.vmul",                  "a5vm.vdiv",
    "a5vm.vaddc",                 "a5vm.vsubc",
    "a5vm.vaddcs",                "a5vm.vsubcs",
    "a5vm.vmax",                  "a5vm.vmin",
    "a5vm.vshl",                  "a5vm.vshr",
    "a5vm.vand",
    "a5vm.vor",
    "a5vm.vxor",
    "a5vm.vsel",                  "a5vm.vselr",
    "a5vm.vselrv2",               "a5vm.vcmp",
    "a5vm.vcmps",                 "a5vm.pdintlv_b8",
    "a5vm.pintlv_b16",
    "a5vm.vtrc",                  "a5vm.vcvt",
    "a5vm.vintlv",                "a5vm.vdintlv",
    "a5vm.vintlvv2",              "a5vm.vdintlvv2",
    "a5vm.vmuls",                 "a5vm.vadds",
    "a5vm.vmaxs",                 "a5vm.vmins",                 "a5vm.vlrelu",
    "a5vm.vshls",                 "a5vm.vshrs",
    "a5vm.vbcnt",                 "a5vm.vcls",
    "a5vm.vmull",                 "a5vm.vmula",
    "a5vm.vsld",                  "a5vm.vldx2",
    "a5vm.vsldb",
    "a5vm.vsts",                  "a5vm.vscatter",
    "a5vm.vsts_pred",             "a5vm.vsst",
    "a5vm.vstx2",                 "a5vm.vsstb",
    "a5vm.vsta",                  "a5vm.vstas",
    "a5vm.vstar",                 "a5vm.pstu",
    "a5vm.vstu",                  "a5vm.vstus",
    "a5vm.vstur",
    "a5vm.psts",
    "a5vm.copy_ubuf_to_gm"};

static bool isSupportedA5VMPrimitive(Operation *op) {
  if (!isa<a5vm::SetFlagOp, a5vm::WaitFlagOp, a5vm::PipeBarrierOp,
           a5vm::GetBufOp, a5vm::RlsBufOp,
           a5vm::SetLoop2StrideOutToUbOp, a5vm::SetLoop1StrideOutToUbOp,
           a5vm::SetLoopSizeOutToUbOp, a5vm::SetLoop2StrideUbToOutOp,
           a5vm::SetLoop1StrideUbToOutOp, a5vm::SetLoopSizeUbToOutOp,
           a5vm::CopyGmToUbufOp, a5vm::CopyUbufToUbufOp,
           a5vm::VldsOp, a5vm::VldasOp, a5vm::VldusOp,
           a5vm::VbrOp, a5vm::VdupOp, a5vm::VciOp,
           a5vm::VbitsortOp, a5vm::Vmrgsort4Op,
           a5vm::Vgather2Op, a5vm::VgatherbOp, a5vm::Vgather2BcOp, a5vm::PsetB8Op,
           a5vm::PsetB16Op, a5vm::PsetB32Op, a5vm::PgeB8Op,
           a5vm::PgeB16Op, a5vm::PgeB32Op, a5vm::PpackOp,
           a5vm::PunpackOp, a5vm::PnotOp, a5vm::PselOp, a5vm::PldsOp,
           a5vm::PldOp, a5vm::PldiOp, a5vm::PstOp, a5vm::PstiOp,
           a5vm::VabsOp, a5vm::VcaddOp,
           a5vm::VcmaxOp, a5vm::VcminOp, a5vm::VexpOp, a5vm::VlnOp,
           a5vm::VsqrtOp, a5vm::VrecOp, a5vm::VreluOp, a5vm::VnotOp,
           a5vm::VaddOp, a5vm::VsubOp, a5vm::VmulOp, a5vm::VdivOp,
           a5vm::VaddcOp, a5vm::VsubcOp, a5vm::VaddcsOp, a5vm::VsubcsOp,
           a5vm::VselOp, a5vm::VselrOp, a5vm::Vselrv2Op, a5vm::VcmpOp, a5vm::VcmpsOp, a5vm::PdintlvB8Op,
           a5vm::PintlvB16Op, a5vm::VtrcOp, a5vm::VcvtOp,
           a5vm::VintlvOp, a5vm::VdintlvOp, a5vm::Vintlvv2Op,
           a5vm::Vdintlvv2Op,
           a5vm::VandOp, a5vm::VorOp, a5vm::VxorOp, a5vm::VshlOp,
           a5vm::VshrOp, a5vm::VbcntOp, a5vm::VclsOp,
           a5vm::VmaxOp, a5vm::VminOp, a5vm::VmulsOp, a5vm::VaddsOp,
           a5vm::VmaxsOp, a5vm::VminsOp, a5vm::VlreluOp, a5vm::VshlsOp,
           a5vm::VshrsOp, a5vm::VmullOp, a5vm::VmulaOp, a5vm::VsldOp,
           a5vm::Vldx2Op, a5vm::VsldbOp, a5vm::VstsOp, a5vm::VscatterOp,
           a5vm::VstsPredOp, a5vm::PstsOp, a5vm::VsstOp, a5vm::Vstx2Op,
           a5vm::VsstbOp, a5vm::VstaOp, a5vm::VstasOp, a5vm::VstarOp,
           a5vm::PstuOp, a5vm::VstuOp, a5vm::VstusOp, a5vm::VsturOp,
           a5vm::CopyUbufToGmOp>(op))
    return false;

  return llvm::is_contained(kSupportedA5VMPrimitives,
                            op->getName().getStringRef());
}

static void collectUnsupportedA5VMOps(Operation *op,
                                      llvm::SmallVectorImpl<std::string> &out) {
  if (op != op->getParentOfType<ModuleOp>() &&
      op->getName().getDialectNamespace() == "a5vm" &&
      !isSupportedA5VMPrimitive(op)) {
    out.push_back(("unsupported-op=" + op->getName().getStringRef()).str());
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nested : block)
        collectUnsupportedA5VMOps(&nested, out);
    }
  }
}

} // namespace

LogicalResult translateA5VMModuleToText(ModuleOp module, llvm::raw_ostream &os,
                                        const A5VMEmissionOptions &options,
                                        llvm::raw_ostream &diagOS) {
  llvm::SmallVector<std::string, 8> unsupportedOps;
  collectUnsupportedA5VMOps(module.getOperation(), unsupportedOps);

  if (failed(writeReportFile(options, unsupportedOps))) {
    diagOS << "A5VM emission failed: could not write unresolved report to '"
           << options.unresolvedReportPath << "'\n";
    return failure();
  }

  if (options.printIntrinsicSelections) {
    diagOS << "A5VM emission note: preserving raw corrected A5VM backend text "
              "at the Phase 1 seam; intrinsic selection is deferred.\n";
  }

  if (!unsupportedOps.empty() && !options.allowUnresolved) {
    for (const std::string &record : unsupportedOps)
      diagOS << "A5VM emission failed: " << record << "\n";
    return failure();
  }

  module->print(os);
  if (!unsupportedOps.empty()) {
    os << "\n";
    for (const std::string &record : unsupportedOps)
      os << "// A5VM-UNSUPPORTED: " << record << "\n";
  }

  return success();
}

} // namespace mlir::pto
