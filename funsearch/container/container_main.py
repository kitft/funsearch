import pickle
import sys


def main(prog_file: str, input_file: str, output_file: str):
  print(f"Running main(): {prog_file}, {input_file}, {output_file}")
  with open(prog_file, "rb") as f:
    func = pickle.load(f)

    with open(input_file, "rb") as input_f:
      input_data = pickle.load(input_f)

      ret = func(input_data)
      with open(output_file, "wb") as of:
        print(f"Writing output to {output_file}")
        pickle.dump(ret, of)


if __name__ == '__main__':
  if len(sys.argv) != 4:
    sys.exit(-1)
  main(sys.argv[1], sys.argv[2], sys.argv[3])
