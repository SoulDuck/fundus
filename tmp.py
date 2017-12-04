import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nargs_int_type', nargs='+', type=int)
args=parser.parse_args()

print args.nargs_int_type