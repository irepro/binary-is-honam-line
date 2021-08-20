import attend_check
import sys

check = attend_check.attend_check()
js = check.detect_member(sys.argv[1])

print(js)
