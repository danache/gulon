
# coding: utf-8

# In[1]:


import pandas as pd
import cv2


# In[2]:


img_name = "ff0025574be642b231354d49556304bd39a9b88a.jpg"


# In[3]:


img_path = "/home/dan/test_img/ff0025574be642b231354d49556304bd39a9b88a.jpg"


# In[4]:


json_file = "/home/dan/test_img/keypoint_train_annotations_20170909.json"


# In[5]:


label = pd.read_json(json_file)


# In[6]:


now_img = pd.Series(label[label['image_id'] == "ff0025574be642b231354d49556304bd39a9b88a"]["keypoint_annotations"])


# In[7]:



for key in now_img.keys():
    dic = now_img[key]
    for key2 in dic.keys():
        b = dic[key2]


# In[8]:


b


# In[25]:


train_lst = open("/home/dan/test_img/train.lst","w")
img_path = "/home/dan/test_img/ff0025574be642b231354d49556304bd39a9b88a.jpg"
img = cv2.imread(img_path)
strs = "0\t2\t3\t750\t500"
#"0\t"+"0\t"+"3\t"#+str(img.shape[1])+"\t"+str(img.shape[0]) + "\t"
print (strs)
num = 0
for i in range(len(b)):
    if i % 3 == 2:
        continue
    if i % 3 == 0:
        strs = strs +"\t"+ str(num) + "\t"
        num += 1
        tmpf = "%.5f" % float(b[i] / img.shape[1])
        strs = strs +tmpf+ "\t"
    else:
        tmpf = "%.5f" % float(b[i] / img.shape[0])
        strs = strs +tmpf
    
strs = strs[:-1]
train_lst.write(strs)
train_lst.write("\t")
train_lst.write(img_path)
train_lst.write("\n")
train_lst.close()
print (strs)

# In[ ]:





# In[26]:


strs


# In[11]:





# In[ ]:




