// SPDX-License-Identifier: LGPL-3.0-or-later
/* ----------------------------------------------------------------------
   LAMMPS plugin registration for the standalone libsog_lmp.so.

   Exports kspace_style sog and fix sog/response.  See
   https://docs.lammps.org/Developer_plugins.html — `plugin load` looks for
   lammpsplugin_init and registers the styles listed here.
------------------------------------------------------------------------- */
#include "fix_sog_response.h"
#include "lammpsplugin.h"
#include "sog.h"
#include "version.h"

using namespace LAMMPS_NS;

static KSpace *sogkspace(LAMMPS *lmp) { return new SOGKSpace(lmp); }
static Fix *fixsogresponse(LAMMPS *lmp, int narg, char **arg) {
  return new FixSOGResponse(lmp, narg, arg);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc) {
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc)regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "kspace";
  plugin.name = "sog";
  plugin.info = "SOG sum-of-Gaussians kspace solver";
  plugin.author = "Zhen Jiang (SJTU)";
  plugin.creator.v1 = (lammpsplugin_factory1 *)&sogkspace;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);

  plugin.version = LAMMPS_VERSION;
  plugin.style = "fix";
  plugin.name = "sog/response";
  plugin.info = "SOG charge-response fix";
  plugin.author = "Zhen Jiang (SJTU)";
  plugin.creator.v2 = (lammpsplugin_factory2 *)&fixsogresponse;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
