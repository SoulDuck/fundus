import os  , sys ,glob

paths=glob.glob('*.png')
for src_path in paths:
    new_path=src_path.replace('.png.png..png' , '.png')
    os.rename(src_path , new_path)
os.rename