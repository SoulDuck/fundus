import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nargs', nargs='+', type=int , default=[1,2,3])
parser.add_argument('--a')
args=parser.parse_args()

print type(args.nargs[0])
print args.nargs
print args.a

