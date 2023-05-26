# %%
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import cv2
import os

big_url   = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
large_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
huge_url  = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"


def download_ckpt(model_name):
    def download_url_(url_path):    
        path = str.split(url_path, " ")[-1]
        name = Path(path).name
        if not os.path.exists(name):
            os.system(f"wget {url_path}")
    
    if model_name == 'b':
        download_url_(big_url)
    elif model_name == 'l':
        download_url_(large_url)
    elif model_name == 'h':
        download_url_(huge_url)
    else:
        print(f"Found Unknown model {model_name}")
        print("Please specify model size: b, l, h")
        return




# %%
# Load SAM model
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def main(args):    
    # Download 
    download_ckpt(args.model)
    
    # Load model
    from segment_anything import sam_model_registry, SamPredictor
    model_type = f"vit_{args.model}"
    checkpoint_path = Path(".")
    
    if torch.cuda.is_available():
        device = "cuda"
    else: device = 'cpu'
    
    ckpt_dict = {'vit_l': 'sam_vit_l_0b3195.pth','vit_b':'sam_vit_b_01ec64.pth', 'vit_h':'sam_vit_h_4b8939.pth'}
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path/ckpt_dict[model_type])
    
    sam.to(device=device)
    predictor = SamPredictor(sam)


    # Load image with cv
    image = cv2.imread(args.image)
    predictor.set_image(image)

    # Get input points
    global img
    global input_point_ls
    global input_label_ls

    input_point_ls = []
    input_label_ls = []
    img = image.copy()
   
   # Display Manual 
    # cv2.putText(img, "L: positive, R: negative, M: reset", (10, 10), "Arial", 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    while True:
        # function to display the coordinates 
        # of the points clicked on the image 
        def click_event(event, x, y, flags, params):
            # Reset
            if event == cv2.EVENT_MBUTTONDOWN:
                global img
                global input_point_ls
                global input_label_ls
                
                img = image.copy()
                input_point_ls = []
                input_label_ls = []
                
                print(input_point_ls)
                print("Reset")
                cv2.imshow('image', img)
                
            # checking for left mouse clicks
            if event == cv2.EVENT_LBUTTONDOWN:
                # displaying the coordinates
                # on the Shell
                print(x, ' ', y)
                input_point_ls.append([x,y])
                input_label_ls.append(1)
                print(input_point_ls)
                
                # displaying the coordinates
                # on the image window
                cv2.drawMarker(img, (x, y), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
                cv2.imshow('image', img)

        
            # checking for right mouse clicks     
            if event == cv2.EVENT_RBUTTONDOWN:
                # displaying the coordinates
                # on the Shell
                print(x, ' ', y)
                input_point_ls.append([x,y])
                input_label_ls.append(0)
                
                # displaying the coordinates
                # on the image window
                cv2.drawMarker(img, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
                cv2.imshow('image', img)


        # displaying the image
        cv2.imshow('image', img)

        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', click_event)

        while True:
            # wait for a key to be pressed to exit
            res = cv2.waitKey(0)
            if res == ord('q'):
                cv2.destroyAllWindows()
                break


        input_point = np.stack(input_point_ls)
        input_label = np.stack(input_label_ls)

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        masks.shape  # (number_of_masks) x H x W


        def click_event(event, x, y, flags, params):
            global clicked_coord
            # checking for left mouse clicks
            if event == cv2.EVENT_LBUTTONDOWN:
        
                # displaying the coordinates
                # on the Shell
                print(x, ' ', y)
                # displaying the coordinates
                # on the image window
                canvas_ = canvas.copy()
                cv2.drawMarker(canvas_, (x, y), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
                cv2.imshow('canvas', canvas_)        
                clicked_coord = [x,y]

        h,w = img.shape[:2]
        canvas_ls = [cv2.resize(img, (w//2, h//2))]

        img = image.copy()
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_image = np.zeros_like(img)
            mask_image[mask] = [255, 144, 30]  # Set color of Mask
            opacity = 0.4 # Set the opacity of the mask
            overlay = cv2.addWeighted(img, 1-opacity, mask_image, opacity, 0)
            
            canvas_ls.append(cv2.resize(overlay, (w//2, h//2)))

        canvas = cv2.vconcat([
            cv2.hconcat([canvas_ls[0], canvas_ls[1]]), 
            cv2.hconcat([canvas_ls[2], canvas_ls[3]])
            ])

        # cv2.putText("Click the image to get the mask", (10, 10), "Arial", 1, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText("Click the First image to go back", (10, 30), "Arial", 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('canvas', canvas)
        cv2.setMouseCallback('canvas', click_event)
        while True:
            # wait for a key to be pressed to exit
            res = cv2.waitKey(0)
            if res == ord('q'):
                cv2.destroyAllWindows()
                break


        x, y = clicked_coord
        if x<w//2 and y<h//2:
            idx = 0
        elif x>=w//2 and y<=h//2:
            idx = 1
        elif x<w//2 and y>h//2:
            idx = 2
        elif x>=w//2 and y>=h//2:
            idx = 3
        else: raise ValueError

        img = image.copy()
        mask_image = np.zeros_like(img)
        mask_image[masks[idx-1]] = [255, 144, 30]  # Set color of Mask
        opacity = 0.4 # Set the opacity of the mask
        overlay = cv2.addWeighted(img, 1-opacity, mask_image, opacity, 0)

        cv2.imshow('overlay', overlay)
        while True:
            # wait for a key to be pressed to exit
            res = cv2.waitKey(0)
            if res == ord('q'):
                cv2.destroyAllWindows()
                break
    
        if idx != 0:
            break
    
    mask = masks[idx-1]
    
    seg_image = img.copy()
    msk = mask.copy()
    
    xmsk = msk != np.roll(msk,1,0)
    ymsk = msk != np.roll(msk,1,1)
    
    msk_edge = np.logical_or(xmsk, ymsk)
    
    # Apply Gaussian blur for smoothing around the edges
    blurred = cv2.GaussianBlur(seg_image, (5, 5), 100)
    seg_image[msk_edge] = blurred[msk_edge]
    seg_image[~msk] = 0

    from PIL import Image
    from rembg import remove
    
    seg_image_pil = Image.fromarray(seg_image)
    
    # Removing the background from the given Image
    finer_image_pil = remove(seg_image_pil)
    
    # Converting the Image back to RGB mode
    finer_image = np.array(finer_image_pil)

    # Display the original image, edges, and smoothed image
    # cv2.imshow("Blurred Image", blurred)
    cv2.imshow("Original Image", img)
    # cv2.imshow("Edges", msk_edge.astype(np.uint8)*255)
    cv2.imshow("Segment Image", seg_image)
    cv2.imshow("Finer Image", finer_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save image with cv2
    cv2.imwrite(f'{args.image}_seg.png', finer_image)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                        prog='Sementation',
                        description='Given image, it runs interactive program to sement image')

    parser.add_argument("--image", "-i", type=str, default="/Users/taerimyoon/Desktop/image2.png")
    parser.add_argument("--model", "-m", type=str, default='h')

    args = parser.parse_args()
    main(args)


# %%
