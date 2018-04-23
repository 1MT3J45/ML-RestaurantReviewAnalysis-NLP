import sys

def load(counter, base, text):
    i = counter
    width = base
    j = int((i+1)/len(width) *100)
    sys.stdout.write(('=' * j) + ('' * (100 - j)) + ("\r%s [ %d" % (text,j) + "% ] "))
    sys.stdout.flush()