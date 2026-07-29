from fermipy.gtanalysis import GTAnalysis

CONFIG = "/Users/maomao/sfsu_research/minniti/configs/minniti_standard_config.yaml"

print(f"Loading config: {CONFIG}")

gta = GTAnalysis(CONFIG, logging={"verbosity": 3})

print("Running gta.setup()...")
gta.setup()

print("Printing ROI...")
gta.print_roi()

print("\nSMOKE TEST FINISHED SUCCESSFULLY")