import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import imageio
import os

global count 
count = 0

global data
data = "november_2019"

global images
images = []
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def showDf(df_select):

  cellColours=plt.cm.RdYlGn(df_select == 0.0)

  shapemat = df_select == 0.0

  colors = np.zeros((shapemat.shape[0], shapemat.shape[1]), dtype=float)

  #print colors

  #print cellColours.shape
  for i in range(colors.shape[0]):
    for j in range(1, colors.shape[1]):
      colors[i,j] = (df_select.iloc[ i , j ] -1 )/10
      #print colors[i,j]
      
  cellColours=plt.cm.Greys(colors)
  #print cellColours.shape
  for i in range(colors.shape[0]):
    for j in range(3, colors.shape[1]):
      if (df_select.iloc[ i , j ] == 0.0):
        cellColours[i,j,:] = [0, 0, 0, 1]
      #colors[i,j] = 1 - df_select.iloc[ i , j ]/10
      #print colors[i,j]
      


  #print cellColours
  table = ax.table(cellText=df_select.values, colLabels=df_select.columns, loc='center', colWidths = [0.5, 0.18, 0.18, 0.10, 0.10, 0.10, 0.10, 0.10, 0.1, 0.1, 0.1, 0.1, 0.1], cellColours=cellColours)

  table.auto_set_font_size(False)
  table.set_fontsize(7)
  table.scale(0.6, 1.4)  # may help
  table.pad = 0.01
  #table.set_text_props(horizontalalignment='left')

  #fig.tight_layout()
  plt.ion()
  plt.show()
  plt.pause(0.001)

  global count
  global data
  plt.savefig(data + '/{}.png'.format(count))

  images.append(imageio.imread(data + '/{}.png'.format(count)))

  count = count+1

def sortDf(df_in, offset):
  df = df_in.copy()

  shapemat = df == 0.0
  shape = shapemat.shape



  value_list = np.zeros((shape[0], 2), dtype=float)



  for i in range(shape[0]):
    total = 0 

    value_list[i,1] = i 
    for j in range(3, shape[1]):
      val = df.iloc[ i , j ]  
      if (val <= 12 and val > 0):

        value_list[i,0] = value_list[i,0] + pow(20, 12 - val)

   
  value_list = value_list[value_list[:,0].argsort()]

  df_sorted = df.copy() #pd.DataFrame(data, columns = ["Titel", "1e Keus", "2e Keus", "Wouter", "Maaike", "Evelien", "Sander", "Isabel", "Willemijn", "Arwen", "Barry", "Mathilde"]) 

  m = 0
  for i in range(shape[0] - 1,  offset, -1):
    k = value_list[i,1].astype(np.int32)
    for j in range(shape[1]):
      df_sorted.iloc[ m , j ] =  df.iloc[ k , j ]  
    m = m + 1

  return df_sorted.copy()

fig, ax = plt.subplots()

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

df = pd.read_csv(data + '/data.csv')
df['1e Keus'] = 0
df['2e Keus'] = 0
df = df.fillna(0)
#df = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))

df_select = df[["Titel"]]


df_select['1e Keus'] = 0
df_select['2e Keus'] = 0

for c in ["Wouter", "Maaike", "Evelien", "Sander", "Isabel", "Willemijn", "Arwen", "Matthijs", "Barry", "Mathilde"]:
  #df_select[c] = df[c].astype('int32')
  df_select[c[:5]] = df[c]

df_select['Titel'] = df_select['Titel'].str.slice(0,20)

#print df_select
#print df_select

showDf(df_select)

showDf(df_select)

shapemat = df_select == 0.0
shape = shapemat.shape



value_list = np.zeros((shape[0], 2), dtype=float)

#print shape
#print value_list


for i in range(shape[0]):
  total = 0 

  value_list[i,1] = i 
  for j in range(3, shape[1]):
    val = df_select.iloc[ i , j ]  
    #print val
    if (val <= 12 and val > 0):


      #print val
      #print i
      value_list[i,0] = value_list[i,0] + pow(20, 12 - val)

 
value_list = value_list[value_list[:,0].argsort()]

#print value_list[-1,1]


#value_list = np.sort(value_list, axis=0, kind='quicksort', order=['f0'])

#print value_list

#print df_select.iloc[23, 0]  

df_sorted = df_select.copy()

m = 0
for i in range(shape[0] - 1, -1 , -1):
  #print "i", i
  k = value_list[i,1].astype(np.int32)
  #print "k", k
  for j in range(shape[1]):
    #print i, j
    #print df_select.iloc[ k , j ]  
    df_sorted.iloc[ m , j ] =  df_select.iloc[ k , j ]  
  m = m + 1


showDf(df_sorted)



for i in range(0, shape[0]):
  df_sorted.iloc[ i , 1 ] = 0
  df_sorted.iloc[ i , 2 ] = 0
  for j in range(3, shape[1]):
    if (df_sorted.iloc[ i , j ] == 1):
      df_sorted.iloc[ i , 1 ] = df_sorted.iloc[ i , 1 ] + 1
    if (df_sorted.iloc[ i , j ] == 2):
      df_sorted.iloc[ i , 2 ] = df_sorted.iloc[ i , 2 ] + 1


showDf(df_sorted)
#print "i", i

for offset in range(0, shape[0]-1):


  #offset = 0

  i = shape[0] - 1 - offset

  #k = value_list[i,1].astype(np.int32)
  #print "k", k
  for j in range(3, shape[1]):
    #print i, j
    #print df_sorted.iloc[ i , j ]  
    val = df_sorted.iloc[ i , j ]  
    if (val > 0):
      for l in range(shape[0] - 1 - offset, -1 , -1):
        if df_sorted.iloc[ l , j ] > val:
          df_sorted.iloc[ l , j ] = df_sorted.iloc[ l , j ] - 1
    df_sorted.iloc[ i , j ] =  0  
  #m = m + 1


  df1 = df_sorted.copy()
  df_sorted = sortDf(df1, offset)

  for i in range(0, shape[0]):
    df_sorted.iloc[ i , 1 ] = 0
    df_sorted.iloc[ i , 2 ] = 0
    for j in range(3, shape[1]):
      if (df_sorted.iloc[ i , j ] == 1):
        df_sorted.iloc[ i , 1 ] = df_sorted.iloc[ i , 1 ] + 1
      if (df_sorted.iloc[ i , j ] == 2):
        df_sorted.iloc[ i , 2 ] = df_sorted.iloc[ i , 2 ] + 1


  showDf(df_sorted)
 # time.sleep(3)

for i in range(0, shape[0]):
  winning_book = df_sorted.iloc[ i , 0 ]
  df_orig = df_select.loc[df_select.Titel == winning_book]

  #print df_orig
  for j in range(3, shape[1]):
    df_sorted.iloc[ i , j ] = df_orig.iloc[0, j]


showDf(df_sorted)
showDf(df_sorted)
showDf(df_sorted)
showDf(df_sorted)
showDf(df_sorted)

#print df_sorted

imageio.mimsave(data + '/result.gif', images, duration=3)


