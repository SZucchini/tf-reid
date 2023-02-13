import glob
import torch
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('../../models/vit_server_cpu.pth')
model.eval()
model.to(device)

img_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

img_path = "../../data/interim/param_test/image_sim/*.jpg"
files = glob.glob(img_path)
files.sort()

img = [img_transforms(Image.open(file).convert("RGB")) for file in files]
img = torch.stack(img).to(device)

with torch.no_grad():
    output = model(img)

print()
for f, o, a in zip(files, output, output.argmax(dim=1)):
    print(f.split('/')[-1], o.size(), a)
