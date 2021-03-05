import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import skimage.io as io
import cv2
import os, os.path
import math
import pygal

from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

###########################################
simulation_frame_data = [] #[{"filename": filename, "img_as_float": img_as_float value},...] เป็น list ที่ครอบ Dict ไว้
simulation_filename = [] #ชื่อไฟล์รูป simulation
realistic_frame_data = [] #[{"filename": filename, "img_as_float": img_as_float value},...] เป็น list ที่ครอบ Dict ไว้
realistic_filename = [] #ชื่อไฟล์รูปจริง
imgs_substraction_data = []
##########################################
imgs_frame_mse = []
imgs_frame_ssim_error = []
imgs_frame_ssim = []

# Data Preparation imread.cvt.gray to im_as_float

# สร้างฟังก์ชั่น เตรียมข้อมูล
def sim_data_prep(path):
    """simulation images"""  
    for file in os.listdir(path):
        img = cv2.imread(path+"//"+file)
        simulation_frame_data.append({
            "filename": file,
            "img_as_float": img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        })
        simulation_filename.append(file)

# สร้างฟังก์ชั่น เตรียมข้อมูล
def real_data_prep(path):
    """realistic images"""
    for file in os.listdir(path):
        img = cv2.imread(path+"//"+file)
        realistic_frame_data.append({
            "filename": file,
            "img_as_float": img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        })
        realistic_filename.append(file)

#find img subtraction of each couple
#สร้างฟังก์ชัน หาผลการลบของข้อมูลรูปภาพทั้งสอง
def find_img_subtraction(sim_frame_data, real_frame_data):
    """image differencing"""
    for i in range(len(sim_frame_data)):
        imgs_substraction_data.append(abs(sim_frame_data[i]["img_as_float"]-real_frame_data[i]["img_as_float"]))

#find ssim amd mse of each imgs
# สร้างฟังก์ชั่น หา ค่า mse
def find_img_mse(sim_frame_data):
    """simulation images"""
    #ควรที่จะเปลี่ยนรายละเอียดฟังก์ชันเพื่อเปรียบเทียบเฟรมจำลอง กับ เฟรมจริง
    for i in range(len(sim_frame_data)):
        imgs_frame_mse.append(float("%.8f"%mean_squared_error(simulation_frame_data[i]["img_as_float"], realistic_frame_data[i]["img_as_float"])))


#สร้างฟังก์ชั่น หาค่า SSIM
def find_img_ssim(real_frame_data):
    for i in range(len(real_frame_data)):
        imgs_frame_ssim.append(float("%.2f"%(ssim(simulation_frame_data[i]["img_as_float"],realistic_frame_data[i]["img_as_float"],data_range=realistic_frame_data[i]["img_as_float"].max()-realistic_frame_data[i]["img_as_float"].min()))))
        imgs_frame_ssim_error.append(abs((imgs_frame_ssim[i]-1)))

# เรียกใช้ฟังก์ชั่นที่ได้ประกาศไว้ร่วมกับ parameter ที่กำหนด funtion_name(parameter)
sim_data_prep('./SIM')
real_data_prep('./REAL')
find_img_mse(simulation_frame_data)
find_img_ssim(simulation_frame_data)
find_img_subtraction(simulation_frame_data, realistic_frame_data)


# Draw and Render to Chart.svg
line_chart = pygal.Line(width=2000)
line_chart.title = 'The Visualization of Images MSE and Images SSIM'
line_chart.x_labels = map(str, range(1, 101))
line_chart.add('MSE', imgs_frame_mse)
line_chart.add('SSIM', imgs_frame_ssim)
line_chart.add('error', imgs_frame_ssim_error)
line_chart.render_to_file('chart.svg')


# Write CSV file Store MSE and SSIM DATA
calculated_data = {
    "Simulation Filename": simulation_filename,
    "Realistic Filename": realistic_filename,
    "Mean Square Error(MSE)": imgs_frame_mse,
    "Structural Similarity(SSIM)": imgs_frame_ssim,
    "ERROR" : imgs_frame_ssim_error,
    "Img Subtraction": imgs_substraction_data,
}# สร้างชุดข้อมูลลักษณะ Dict ขึ้นมา โดยเพิ่มข้อมูลที่ต้องการลงไปในแต่ละคอลัม หัวตาราง: ข้อมูล,ในแต่ละคอลัม
df = pd.DataFrame(calculated_data, columns= ["Simulation Filename", "Realistic Filename", "Mean Square Error(MSE)", "Structural Similarity(SSIM)", "ERROR", "Img Subtraction"])# สร้าง Dataframe ด้วยการเรียกชุดข้อมูล calculated_data ที่ได้สร้างไว้ก่อนหน้า
df.to_csv ('calculated_data.csv', index = False, header=True)# สร้างไฟล์ csv จาก dataframe และชุดข้อมูลข้างบน
print(df)# ทดลองปริ้นแสดงผลข้อมูลในตาราง

rows, cols = simulation_frame_data[0]["img_as_float"].shape #กำหนดขนาดของแต่ละคอลัมและแถวโดยใช้ขนาดของข้อมูล

fig, axes = plt.subplots(len(simulation_frame_data), 3, figsize=(12, 300),sharex=False, sharey=True) # กำหนดขนาดของ axes และ figure พร้อมใช้จำนวน แถว*คอลัม ขนาด = จำนวนข้อมูล*3

label = 'MSE: {:.8f}, SSIM: {:.2f}' #กำหนด label แสดงผลในแต่ละ figure


# ทำการ plot ข้อมูลลงในตำแหน่งทีต้องการต่างๆ
for i in range(len(simulation_frame_data)):
  
    axes[i, 0].imshow(np.random.rand(8, 90), interpolation='nearest', aspect="auto") # กำหนดคุณภาพของรูปภาพที่จะแสดงให้ไม่เบลอ
    axes[i, 0].imshow(simulation_frame_data[i]["img_as_float"], cmap=plt.cm.gray, vmin=0, vmax=1 ) # แสดงภาพด้วยช้อมูลที่ทำการสร้างไว้
    axes[i, 0].set_xlabel(label.format(imgs_frame_mse[i], imgs_frame_ssim[i])) # กำหนดเลเบลแสดงข้อความโดยอ้างอิงจาก label
    axes[i, 0].set_title(simulation_frame_data[i]["filename"]) # กำหนด title ให้แสดงผลเป็นชื่อของภาพ

    axes[i, 1].imshow(np.random.rand(8, 90), interpolation='nearest', aspect="auto")
    axes[i, 1].imshow(realistic_frame_data[i]["img_as_float"], cmap=plt.cm.gray, vmin=0, vmax=1 )
    axes[i, 1].set_xlabel(label.format(imgs_frame_mse[i], imgs_frame_ssim[i]))
    axes[i, 1].set_title(realistic_frame_data[i]["filename"])

    axes[i, 2].imshow(np.random.rand(8, 90), interpolation='nearest', aspect="auto")
    axes[i, 2].imshow(imgs_substraction_data[i], cmap=plt.cm.gray, vmin=0, vmax=1 )
    axes[i, 2].set_xlabel("Subtracted from %s and %s"%(simulation_frame_data[i]["filename"], realistic_frame_data[i]["filename"]))
    axes[i, 2].set_title("Subtraction no. %d"%i)

# fig2,axes2 = plt.subplots(1, 3, figsize=(12, 8),sharex=False, sharey=False)
# for i in range(len(simulation_frame_data)):
#     axes2[0].imshow(np.random.rand(8, 90), interpolation='nearest', aspect="auto") # กำหนดคุณภาพของรูปภาพที่จะแสดงให้ไม่เบลอ
#     axes2[0].imshow(simulation_frame_data[i]["img_as_float"], cmap=plt.cm.gray, vmin=0, vmax=1 ) # แสดงภาพด้วยช้อมูลที่ทำการสร้างไว้
#     axes2[0].set_xlabel(label.format(imgs_frame_mse[i], imgs_frame_ssim[i])) # กำหนดเลเบลแสดงข้อความโดยอ้างอิงจาก label
#     axes2[0].set_title(simulation_frame_data[i]["filename"]) # กำหนด title ให้แสดงผลเป็นชื่อของภาพ

#     axes2[1].imshow(np.random.rand(8, 90), interpolation='nearest', aspect="auto")
#     axes2[1].imshow(realistic_frame_data[i]["img_as_float"], cmap=plt.cm.gray, vmin=0, vmax=1 )
#     axes2[1].set_xlabel(label.format(imgs_frame_mse[i], imgs_frame_ssim[i]))
#     axes2[1].set_title(realistic_frame_data[i]["filename"])

#     axes2[2].imshow(np.random.rand(8, 90), interpolation='nearest', aspect="auto")
#     axes2[2].imshow(imgs_substraction_data[i], cmap=plt.cm.gray, vmin=0, vmax=1 )
#     axes2[2].set_xlabel("Subtracted from %s and %s"%(simulation_frame_data[i]["filename"], realistic_frame_data[i]["filename"]))
#     fig2.savefig('./result/result%s.png'%i)


# allow scrollable windows
class ScrollableWindow(QtWidgets.QMainWindow):
    def __init__(self, fig):
        self.qapp = QtWidgets.QApplication([])

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(10,10,10,10)
        self.widget.layout().setSpacing(10)

        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)

        self.show()
        exit(self.qapp.exec_())

a = ScrollableWindow(fig) # เรียกใช้งานฟังก์ชันอนุญาติให้สามารถเลื่อนหน้าจอได้
plt.subplots_adjust(top = 0.99, left=0.006, Right=0.994, bottom=0.002, hspace=0.368, wspace=0.0)
# plt.tight_layout(h_pad=100, w_pad=100) # แสดงผลแบบ tight layout
plt.show()

