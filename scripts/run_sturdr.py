from sturdr import SturDR

if __name__ == "__main__":
    yaml_filename = "./config/thesis_sim_rcvr.yaml"

    sdr = SturDR(yaml_filename)
    sdr.Start()
