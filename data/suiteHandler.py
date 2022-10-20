import glob, json
import os
from pathlib import Path

''' Train Image Picker
Description : Meta가 존재하는 데이터의 이름을 불러와, 그 이름에 해당하는 이미지를 원하는 위치로 옮겨주는 메소드
Usage       : 전체 데이터에서 일부만 라벨링된 경우 사용. 

Param       -> image_list : 파일 명만 가지고 있어야 함.(경로명 포함 X)

Considering : option = 'mv' or 'cp'
'''
# class Train_Image_Picker:
#     def __init__(self, image_list:Optional[list], from_path:str, to_path:str) -> None:
#         self.image_list = image_list
#         self.from_path = from_path
#         self.to_path = to_path
    
#     def get_imgList(self, meta_path:str):
#         self.image_list = os.listdir(meta_path)

        
#     def move_img(self):
#         assert self.image_list # self.image_list == None 이면 error
        
#         # for l in self.image_list:
#             # path
            
# if __name__=='__main__':
#     tip = Train_Image_Picker()
#     tip.

####################################################################################
# train imge picker
meta_path = '/home/hwi/Downloads/VQIS-POC_(BOX) 2022-10-20 9_27_13 /meta/VQIS/images'

# 하위 모두 가는코드 추가
image_list = os.listdir(meta_path)
image_list = [Path(img).stem for img in image_list]
print(len(image_list))
print(image_list[1])

# mv to to path
from_path = '/home/hwi/github/data/VQIS_PoC/suite'
to_path = '/home/hwi/Downloads/filetered chick'

exits_list = os.listdir(from_path)

for i in image_list:
    if i in exits_list:
        from_ = os.path.join(from_path, i)
        to_ = os.path.join(to_path, i)
        os.rename(from_, to_)

'''
import os
import shutil

os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
'''
####################################################################################

####################################################################################
# suite2csv , bounding box , MinMax
# class_id center_x center_y width height

# path = '/home/hwi/github/data/VQIS-POC_(BOX) middle chick'

# annotation_path='/home/hwi/github/data/middle_chick_detect'

# origin_width = 1520
# origin_height = 2048

# def get_meta_files():
#     files = glob.glob(path+'/meta/**/*.json', recursive=True)
#     return files

# def get_name(file):
#     return Path(Path(file).stem).stem         # delete '.json'

# files = get_meta_files()
# # print(len(files))

# for i, file in enumerate(files):
#     # open metafile
#     with open(file, encoding='UTF8') as json_meta_file:
#         json_meta_data = json.load(json_meta_file)
    
#     # get file name
#     name = get_name(file)

#     # open label file
#     label_path = os.path.join(path, json_meta_data['label_path'][0])
#     with open(label_path, encoding='UTF8') as json_label_data:
#         json_label_data = json.load(json_label_data)
#     print(json_label_data)
#     print()
#     print('name', name)
#     print()
    
#     result = ''

#     # get 육계
#     # if label exists on data
#     if 'objects' in json_label_data:
#         # object_num = len(json_label_data['objects'])
#         # for num in range(object_num):
#         #     print(num)
    
#         for obj in json_label_data['objects']:
#             if obj['class_name'] == '육계':
#                 # print('yes')
#                 print(obj['annotation']['coord'])
#                 x = obj['annotation']['coord']['x']
#                 y = obj['annotation']['coord']['y']
#                 width = obj['annotation']['coord']['width']
#                 height = obj['annotation']['coord']['height']
#                 print(x, y, width, height)
                
#                 x_center = (x+(width/2)) / origin_width
#                 y_center = (y+(height/2)) / origin_height
#                 width = width / origin_width
#                 height = height / origin_height

#                 print(x_center, y_center, width, height)
#                 result +=f'0 {x_center} {y_center} {width} {height}\n'
#                 print()
            
#             output_name = os.path.join(annotation_path, name+'.txt')
#             with open(output_name, 'w', encoding='utf-8') as f:
#                 f.write(result)
        
####################################################################################
