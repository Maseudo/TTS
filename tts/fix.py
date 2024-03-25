file = open("soundbites.csv", "r")
output = open("soundbites_fixed.csv", "w")
output_buffer = ""
line = file.readline()

while line != "":
                if len(line) > 90:
                               output_buffer += line
                else:
                                print("Removing line: "+line)
                line = file.readline()

output.write(output_buffer)                