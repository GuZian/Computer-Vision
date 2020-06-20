#Moving Object Detection
#by Zian Gu
#06/18/2020
import cv2
#初始化
camera = cv2.VideoCapture(0)#0表示打开笔记本内置摄像头
avg = None

while True:
    grabbed, frame = camera.read()
    if not grabbed:
        break

    frame = cv2.resize(frame,(900,700))  
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (23, 23), 0)#0是标准差，若该参数是0则根据高斯矩阵的尺寸自己计算

    if avg is None:
       avg = gray.copy().astype("float")#初始化平均帧
       continue
    avg = cv2.accumulateWeighted(gray, avg, 0.5)#更新背景，叠加于avg
    gray = cv2.absdiff(gray, cv2.convertScaleAbs(avg))#帧差法计算两帧的差的绝对值
    
    ret, thresh = cv2.threshold(gray,5,255,cv2.THRESH_BINARY) #对叠加后的灰度图进行二值化处理
    thresh = cv2.dilate(thresh, None, iterations=3)#形态学膨胀处理
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#寻找轮廓（第二个返回参数是层级，实际无用处）
    #cv2.drawContours(frame,contours,-1,(0,0,255),-1)#用红色把变化区域轮廓标记出来（填充）
    #用绿色的框把大幅度运动的区域框出来
    for c in contours:
        if cv2.contourArea(c) > 2000:#降低干扰
            (x, y, w, h) = cv2.boundingRect(c)#外接最小矩形
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Contours", frame)
    cv2.imshow("Gray", gray)
    cv2.imshow("Binary", thresh)
    if cv2.waitKey(20) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
