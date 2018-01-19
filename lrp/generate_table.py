import os
import sys

img_paths = []
heat_paths = []
heat_paths2 = []

labels = []
predictions = []

model_name = "mnist_1_wrong"

for root, dirs, files in os.walk("./out/" + model_name, topdown=False):
   for name in files:
        if name.endswith("png"):
            nums = name.split('-')
            label = nums[1]
            prediction = nums[5]
            fname = name.split('.')[0]
            path = os.path.join(root, name)
            if fname.endswith('norm'):
                img_paths.append(path)
                labels.append(label)
                predictions.append(prediction)
            if fname.endswith('heat'):
                heat_paths.append(path)
            if fname.endswith('heatn'):
                heat_paths2.append(path)

with open("tables/" + model_name + ".html", "w") as of:

    of.write('<div class="modelinfos"></div>\n')

    of.write('<table class="main-table">\n')
    of.write('<tr class="h">')
    of.write('<th>Label</th>')
    of.write('<th>Prediction</th>')
    of.write('<th>Image</th>')
    of.write('<th>Heatmap1</th>')
    of.write('<th>Heatmap2</th>')
    of.write('</tr>')

    for index in range(0, len(labels)):
        of.write('<tr>')
        of.write('<td>' + labels[index] + '</td>')
        of.write('<td>' + predictions[index] + '</td>')
        of.write('<td><img src="' + img_paths[index] + '"></td>')
        of.write('<td><img src="' + heat_paths[index] + '"></td>')
        of.write('<td><img src="' + heat_paths2[index] + '"></td>')
        of.write('</tr>')
    of.write("</table></html>")