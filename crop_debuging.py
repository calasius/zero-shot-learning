import cv2



to_image = transforms.Compose([transforms.ToPILImage(), transforms.Resize(448)])

transform = transforms.Compose([transforms.Resize((450,450)), transforms.CenterCrop(448), transforms.ToTensor()])

def get_attention_map_image(att):
    min_value = torch.min(att)
    max_value = torch.max(att)
    att = att / torch.max(att)
    gray_scale_att_image = Image.fromarray(np.uint8(att.detach().numpy()[0] * 255) , 'L').resize((448,448), Image.BILINEAR)
    heatmapimg = np.array(np.array(gray_scale_att_image), dtype = np.uint8)
    
    return Image.fromarray(cv2.applyColorMap(-heatmapimg, cv2.COLORMAP_JET))
            
            
def plot_heatmap_on_image(image_file, attention_map):
    img = Image.open(image_file)
    img_tensor = transform(img)
    img = TF.to_pil_image(img_tensor)
    att_image = get_attention_map_image(attention_map)
    return Image.blend(img, att_image, 0.3)

            
def plot_box_on_image(box, attention_map, file):
    x = box[0]
    y = box[1]
    l = box[2]
    x = x - l/2
    y = y - l/2
    image = plot_heatmap_on_image(image_file, attention_map)
    fig, ax = plt.subplots()
    ax.imshow(image)
    rect = patches.Rectangle((x,y),w,h, edgecolor='b', facecolor="none")
    ax.add_patch(rect)
    plt.savefig(file)