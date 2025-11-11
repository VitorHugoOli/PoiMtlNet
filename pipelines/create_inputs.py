from configs.paths import IO_CHECKINS
from etl.create_input import create_input

if __name__ == "__main__":
    # Define the path to the input file and the state name
    input_file = IO_CHECKINS / 'Florida.csv'
    state_name = 'florida_dgi_new'

    # Create embeddings for the specified state
    # create_embeddings(state_name, input_file)
    create_input(state_name)