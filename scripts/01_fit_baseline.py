from fermipy.gtanalysis import GTAnalysis

CONFIG = "/Users/maomao/sfsu_research/minniti/configs/minniti_standard_config.yaml"

# Setup
gta = GTAnalysis(CONFIG, logging={"verbosity": 3})
gta.setup()

# First optimize step
gta.optimize()

gta.free_source("galdiff")
gta.free_source("isodiff")

# Free bright source norms anywhere in model
gta.free_sources(minmax_ts=[25, None], pars="norm")

fit = gta.fit()

print("Baseline fit result:")
print(fit)

gta.print_roi()

gta.write_roi("baseline_standard")

print("\nBASELINE SAVED AS baseline_standard")