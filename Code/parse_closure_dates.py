import csv

f = open("../Data/closure_paper_dates.csv")
rd = csv.reader(f)

data = []
for line in rd:
  data.append(line)
header = data[0]

for line in data[1:]:
  sfile = open("../Data/Derived State Data/{}.csv".format(line[0]), "w")
  swriter = csv.writer(sfile)
  contents = []
  for i in range(1,len(line)):
    if line[i] == "N/A":
      continue
    contents.append([line[i].strip(), header[i], "Incidence {}".format(header[i]),"https://jamanetwork.com/journals/jama/fullarticle/2769034?utm_campaign=articlePDF&utm_medium=articlePDFlink&utm_source=articlePDF&utm_content=jama.2020.14348"])
  swriter.writerow(["Dates", "Opening/Closure", "Notes", "Link"])
  swriter.writerows(contents)
  sfile.close()
f.close()
