import matplotlib.pyplot as plt
import sys
import json
import numpy as np

print(list(range(len([9,3,2]))))
def get_data(eval_name):
  with open('evaluations/{}.json'.format(evaluation), 'r') as infile:
    data = json.load(infile)


  epochs = sorted(data.keys(), key=int)
  print(epochs)

  mins = []
  maxs = []
  means = []

  for epoch in epochs:
    mins.append(min(data[epoch]))
    maxs.append(max(data[epoch]))
    means.append(np.mean(data[epoch]))
  return mins, means, maxs

if __name__ == "__main__":
  full_eval = len(sys.argv) == 1

  if not full_eval:
    evaluation = sys.argv[1]
    mins, means, maxs = get_data(evaluation)
    plt.plot(epochs, mins, label='min')
    plt.plot(epochs, maxs, label='max')
    plt.plot(epochs, means, label='mean')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('evaluations/{}.png'.format(evaluation))
  else:
    from glob import glob
    import re

    patt = './evaluations/*.json'
    evaluations = [re.findall(r".*/(.*)\.json", w)[0] for w in glob(patt)]
    #evaluations = ["lr_5e-5_soft_1e-2", "lr_5e-5_hard_2.5e4", "sygnowski", "lr_5e-6_soft_1e-3"]
    #evaluations = ["sygnowski"]
    plt.figure(figsize=(10,5))

    for evaluation in evaluations:
      mins, means, maxs = get_data(evaluation)
      plt.plot(list(range(len(means))), maxs, label=evaluation)

    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('evaluations/all-max.png')
    # Every filename should have a contain a number denoting the step
    # ws = sorted(glob(patt), key=lambda fn: int(re.findall(r"(\d+)\.h5f", fn)[0]))
