class A:
    def __init__(self):
        self.tmp_1()
    def tmp(self):
        print 'tmp'

    def tmp_1(self):
        self.tmp()

a=A()