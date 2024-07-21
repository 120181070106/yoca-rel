```相对的改进
#---------------------------(predict.ipynb)------------------------#
    if mode == "predict":
        image = Image.open('4.jpg')#先自动对目录下的4.jpg文件实施基线预测
        #此外还提供的基准图片有：45是基线目标，67是大目标，89是小目标，cd是难目标
        r_image = yolo.detect_image(image, crop = crop, count=count)
        r_image.show()
        while True:
            img = input('Input image filename:')
            try:#这样自动叠加后缀就只需要输入文件名
                image = Image.open(img+'.jpg')
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()
#---------------------------(yolo.py)------------------------#
        "model_path"        : 'model_data/b基础633.pth',#原yolov8_s换为自训的基线权
        "classes_path"      : 'model_data/voc_classes.txt',#只含0到6七类，分别分行
        "phi"               : 'n',#版本从s换为更易训、内存更小的n 
        "cuda"              : False,#cuda换为否方便推理时切无卡模式用cpu更省钱
#---------------------------(utils_fit.py)------------------------#
    if local_rank == 0:#去掉开训和完训，以及验证全程的显示
        # print('Start Train')
    if local_rank == 0:
        pbar.close()
        # print('Finish Train')
        # print('Start Validation')
        # pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    if local_rank == 0:
        pbar.close()
        # print('Finish Validation')
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):#关掉最优权的保存提示，将定期权重名改为p030三个数的形式，忽略具体损失，最后精简best_epoch_weights为b，last_epoch_weights为l
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            # print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "b.pth"))
        #     torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        # torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
        torch.save(save_state_dict, os.path.join(save_dir, "l.pth"))
#---------------------------(callbacks.py)------------------------#
            # print("Calculate Map.")
            # print("Get map done.") #关掉算map始末的提示
#---------------------------(train.ipynb)------------------------#
if __name__ == "__main__": #精简参数行，去除多余注释
    Cuda            = True #服务器训练只能用gpu，无卡模式cpu训不了
    seed            = 11
    distributed     = False
    sync_bn         = False
    fp16            = True #设true更快些
    classes_path    = 'model_data/voc_classes.txt'
    model_path      = 'b基础633.pth' #原为'model_data/yolov8_s.pth'改成咱们自训的
    input_shape     = [640, 640]
    phi             = 'n' # 原's'改更小更高效
    pretrained      = False #有权重就不用预训练
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    label_smoothing     = 0
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 2 #原32改小
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 4 #原16改小
    Freeze_Train        = False #预冻结前50的骨网权重，在前置网需要同时训练故设False
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.937
    weight_decay        = 5e-4
    lr_decay_type       = "cos"
    save_period         = 30 #每隔30轮保存下权重，整个只需10个文件，减少原10的冗余
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 10
    num_workers         = 4
#---------------------------(voc_annotation.py)------------------------#
annotation_mode     = 2 #基本的集合已被划分于ImageSeg文件夹，现只需生成2007_train.txt、2007_val.txt的目标信息即可（原为0）
#-----------------------------------(utils_map.py)------------------------------#
# 第241和第609行，均加入".manager"变为fig.canvas.manager.set_window_title
    fig.canvas.manager.set_window_title(window_title)#第241行
                fig.canvas.manager.set_window_title('AP ' + class_name)#609行
```
关系框部分（照autodl调出的默认参数为1.5和0.1，但featurize上更为1.3和0.2）
```
#---------------------------------(utils_bbox.py)---------------------------#
# TORCH_1_10 = check_version():随即插入如下↓
def cal_angle(b1, b2, box): # 算两相邻框与当前框之间的夹角
    cen1=(b1[0:2]+b1[2:])/2; cen2=(b2[0:2]+b2[2:])/2; cen=(box[0:2]+box[2:])/2
    angle=torch.dot(cen1-cen,cen2-cen)/(torch.norm(cen1-cen)*torch.norm(cen2-cen))
    return torch.acos(angle)*180/3.1415926  # 转换为角度制
def box(m,w,h):return torch.tensor([m[0]-w/2,m[1]-h/2,m[0]+w/2,m[1]+h/2])
def mid(a框,b框,框数=1):
    cen1=(b框[0:2]+b框[2:])/2; cen2=(a框[0:2]+a框[2:])/2
    m=(cen1+cen2)/2; m1=(2*cen1+cen2)/3; m2=(cen1+2*cen2)/3
    w=(b框[2]-b框[0]+a框[2]-a框[0])/2; h=(b框[3]-b框[1]+a框[3]-a框[1])/2
    return box(m,w,h)  if 框数==1 else [box(m1,w,h),box(m2,w,h)]
def off(a框,b框):return b框-torch.tensor([(a框[0]+a框[2])/2-(b框[0]+b框[2])/2,(a框[1]+a框[3])/2-(b框[1]+b框[3])/2]).repeat(2).to(a框.device)#b框表向当前框偏移
def 关框生成(框集,新框集):
    for i in range(len(框集)):#遍历单图纵所含的[nb,4]必然框
        当前框 = 框集[i]; 距集 = []
        for j in range(len(框集)): # 除了当前框的各框均与当前框算中心点L2距离
            距集.append(torch.norm((框集[j]-当前框)[0:2]+(框集[j]-当前框)[2:])/2)
        if len(框集)==2:return [mid(框集[1-i],当前框,1),off(框集[1-i],当前框)]
        if len(框集)<2: return []#什么都不用返,上面只需返靠向当前框这边的外框
        两短距, 索引 = torch.topk(torch.tensor(距集), k=3, largest=False)
        首近号=索引[1];次近号=索引[2];比例=两短距[2]/两短距[1]#排掉排第一的当前框自己
        a,b=mid(框集[首近号],当前框,2)
        if cal_angle(框集[首近号],框集[次近号],当前框)>90: #连线角为钝才与次框互动
            if 比例>=1.5:新框集.append(mid(框集[次近号],当前框,1))#次框远间生一关框
            if 比例>=2.7:新框集.append(a);新框集.append(b)#多则再两
        else:#两距相近说明两均歧当框,只生一关;特别近说明受完全孤立当前框生两关
            if 比例<1.4: 新框集.append(mid(框集[首近号],当前框,1))
            if 比例<1.1:新框集.append(a);新框集.append(b)#存在多个值时可用加列表
            新框集.append(off(框集[首近号],当前框)) # 向孤立的自己这边偏
    去重框集=[torch.tensor(框) for 框 in set(tuple(框.tolist()) for 框 in 新框集)]
    return 去重框集#内张量元素转列表后再转元组,成不变量以转集合去重
def 未框去误(未框,必框,误框号集):
    if len(必框)>1:
        for i in range(len(未框)):
            当前框 = 未框[i]; 距集 = []
            for j in range(len(必框)): # 除了当前框的各框均与当前框算中心点L2距离
              距集.append(torch.norm((必框[j]-当前框)[0:2]+(必框[j]-当前框)[2:])/2)
            两短距, 索引 = torch.topk(torch.tensor(距集), k=2, largest=False)
            if cal_angle(必框[索引[1]],必框[索引[0]],当前框)<33:
                if 两短距[1]/两短距[0]<1.5:误框号集.append(i)
        return torch.tensor(误框号集).long()
#插上后的126行先更原"non_max_suppression"为"non_max_suppression1"，再插如下函↓
    def non_max_suppression(self,全预测,种类数,input_shape,image_shape,letterbox,conf_thres=0.5,nms_thres=0.4,初阈值=0.5):#[bs,8400,85]预测先转左上右下
        a=全预测[:,:,0]-全预测[:,:,2]/2; b=全预测[:,:,1]-全预测[:,:,3]/2
        c=全预测[:,:,0]+全预测[:,:,2]/2; d=全预测[:,:,1]+全预测[:,:,3]/2
        全预测[:,:,:4]=torch.stack((a,b,c,d),-1); 单图预测=全预测[0]; 出=None
        类信度,似然类=torch.max(单图预测[:,4:4+种类数],1,keepdim=True)#均[8k,1]
        单图预测 = torch.cat((单图预测[:,:4],类信度.float(),似然类.float()),1)
        必然预测=单图预测[(类信度[:,0]>=初阈值).squeeze()]#初阈高点以减少首轮的算量
        必框=必然预测[:,:4].to(全预测.device); 必度=必然预测[:,4]
        关框=关框生成(必框[nms(必框,必度,nms_thres)],[])#必框稀化后生成关系框
        置信预测 = 单图预测[(类信度[:,0]>=0.5*conf_thres).squeeze()]#降阈给低信升机
        未然索引=torch.Tensor(torch.where(置信预测[:,-2]<0.6)[0].float()).long()#[0]转索引元组为张，后面1.9或换成1.5，反正在左右调试,此处为避报错强更为长整
        未框=置信预测[未然索引][:,:4]; 贴矩=torch.zeros((len(未框),len(关框)))
        for i, a in enumerate(未框): #用k代替i防止与前面的i冲突!
            for j, b in enumerate(关框):#作差求绝和得两框集[-,4]贴合度[a,b]
                贴矩[i,j]=torch.abs(a-b.to(a.device)).sum()/(a[2:]-a[:2]).sum()
        if len(关框)>0:置信预测[未然索引[torch.min(贴矩,dim=-1)[0]<1.4],-2]*=1.5#.9
        未然索引=torch.Tensor(torch.where(置信预测[:,-2]<0.3)[0].float()).long()
        未框=置信预测[未然索引][:,:4];误框号集=未框去误(未框,必框[nms(必框,必度,nms_thres)],[]); 置信预测[未然索引[误框号集],-2]*=0.1#必框nms后才有实际意义!
        置信预测=置信预测[置信预测[:,-2]>=conf_thres]
        for c in 置信预测[:,-1].unique(): #取所含所有类得分筛后全部的预测结果
            当类预测=置信预测[置信预测[:, -1] == c]
            当类预测=当类预测[nms(当类预测[:,:4],当类预测[:,4],nms_thres)]#官方快
            出=当类预测 if 出 is None else torch.cat((出,当类预测))
        if 出 is not None:#(x1,y1,x2,y2,信度,类别)叠在之前类的输出上
            出=出.cpu().numpy(); 出[:, :4]=self.yolo_correct_boxes((出[:,0:2]+出[:,2:4])/2,出[:,2:4]-出[:,0:2],input_shape,image_shape,letterbox)
        return [出]#打包成列表,因为源代码以为是可以并行处理的,实际仅一图
#-------------------------------(map.ipynb)---------------------------#
    if map_mode == 4: #如已经"pip install pycocotools"下载过coco工具箱，则此倒数第四个有效行注释掉，以便输出本地的各类结果后随即按coco方式得到各评价指标值
#-------------------------(yolo.py，用于显示)---------------------------#
        "model_path"        : 'model_data/b基础633.pth',
        "model_path"        : 'k.pth',#可再换成此自训的不高不低的权值，前权过精难提
```