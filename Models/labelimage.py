import utils as u
import os

dir_r = 'Data'
dir_c = 'SkinLesion'
dir_ts = 'Testing'
dir_tr = 'Training'
classess = ['akiec', 'bcc', 'mel']
classesc = [
    'High squamous intra-epithelial lesion',
    'Low squamous intra-epithelial lesion',
    'Negative for Intraepithelial malignancy',
    'Squamous cell carcinoma',
]
classesb = ['notumor', 'glioma', 'meningioma']

i = 'images'
for j in range(len(classess)):
    diri = os.path.join(dir_r, dir_c, dir_ts, classess[j], i)
    print(diri)
    l = 'labels'
    dirl = os.path.join(dir_r, dir_c, dir_ts, classess[j], l)
    print(dirl)
    u.create_yolo_labels(diri, dirl, j)
