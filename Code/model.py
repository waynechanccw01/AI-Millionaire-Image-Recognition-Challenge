import torch
import torchvision.transforms as tt
import PIL
from PIL import Image, ImageDraw

from facenet import InceptionResnetV1, MTCNN, SeNet, GanNet

def input_tolist(input_):
    if not isinstance(input_, (list, tuple)):
        input_ = [input_]
    return input_

def convert(input_):
    if isinstance(input_, str):
        input_ = Image.open(input_)
    if isinstance(input_, PIL.Image.Image):
        return input_
    else:
       print('Input type shd be \"path(str)\" or \"PIL.Image.Image\".')

def cal_face_tensor(cnn, input_, prob_threshold=0.98):
    input_ = convert(input_)
    fts, probs = cnn(input_, return_prob=True)
    if fts is not None:
        if fts.dim() == 3:
            fts = fts.unsqueeze(0)
        else:
            fts = [ft.unsqueeze(0) for ft, prob in zip(fts, probs) if prob>prob_threshold]
            fts = torch.cat(fts)
    else:
        fts = torch.ones(1, 3, 160, 160).to(cnn.device)
    return fts

def cal_embs_dist(embs_1, embs_2, dist_intial=100):
    r1 = embs_1.shape[0]
    c2 = embs_2.shape[0]

    dist = torch.ones(r1, c2) * dist_intial

    for i in range(r1):
        for j in range(c2):
            emb1 = embs_1[i].unsqueeze(0)
            emb2 = embs_2[j].unsqueeze(0)
            dist[i][j] = (emb1 - emb2).norm().item()

    return dist

def mt_overlap(dist):
    mean, std = dist.mean(), dist.std()
    margin = mean - std
    row_mins = dist.argmin(dim=1)
    col_mins = dist.argmin(dim=0)

    a = set()
    b = set()
    for i, j in enumerate(row_mins):
        if dist[i][j] < margin:
            a.add((i, j.item()))
    for j, i in enumerate(col_mins):
        if dist[i][j] < margin:
            b.add((i.item(), j))

    overlap = []
    for _ in a.intersection(b):
        overlap.append(_[1])
    return overlap

def img_draw_with_box(img, box):
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    draw.rectangle(box.tolist(), width=5)
    return img_draw

class AI_Model():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.cnn = MTCNN(device=self.device, keep_all=True)
        self.cnn0 = MTCNN(device=self.device)
        self.rnn = InceptionResnetV1(device=self.device, pretrained='vggface2').eval()
        self.senet1 = SeNet(pretrained='senet_1').eval()
        self.senet2 = SeNet(pretrained='senet_2').eval()
        self.senet3 = SeNet(pretrained='senet_3').eval()
        self.gan1 = GanNet(pretrained='gan_1').eval()
        self.gan2 = GanNet(pretrained='gan_2').eval()

    @torch.no_grad()
    def compare(self, input_1, input_2, type_=None, show_imgs=False, save_path='test/'):
        
        if type_ == "S001":
            cnn1 = self.cnn0
            cnn2 = self.cnn0
            rnn = self.rnn
            dist = self.compare_(cnn1, cnn2, rnn, input_1, input_2)
            dist = dist.mean(dim=0)
            ans = dist.argmin().item()

        elif type_ == "S002":
            cnn1 = self.cnn
            cnn2 = self.cnn0
            rnn = self.rnn
            dist = self.compare_(cnn1, cnn2, rnn, input_1, input_2)
            min_ind = dist.argmin()
            ans = min_ind % dist.shape[1]
            ans = ans.item()

        elif type_ == "S003":
            cnn1 = self.cnn0
            cnn2 = self.cnn0
            rnn = self.rnn
            dist = self.compare_(cnn1, cnn2, rnn, input_1, input_2)
            # ans = dist.argmin().item()
            min2 = torch.topk(dist, 2, largest=False).indices.tolist()[0]
            input_2_ = [input_2[_] for _ in min2]
            dist_ = self.checkgan(input_2_)
            ans = min2[dist_.argmin().item()]
        
        elif type_ == "S004":
            cnn = self.cnn0
            dist = self.detect_gender(cnn, input_1) # list
            votes = torch.cat(dist).argmax(1)
            ans = votes.mode().values.item()

        elif type_ == "M001":
            cnn1 = self.cnn
            cnn2 = self.cnn0
            rnn = self.rnn
            dist = self.compare_(cnn1, cnn2, rnn, input_1, input_2)
            ans = mt_overlap(dist)

        elif type_ == "D001":
            cnn1 = self.cnn0
            cnn2 = self.cnn
            rnn = self.rnn
            dist = self.compare_(cnn1, cnn2, rnn, input_1, input_2)
            ind = dist.argmin().item()
            ans = self.show_eyes_xy(cnn2, input_2, ind)
        
        else:
            print("Please select correct \"type\".")
            return False

        return ans

    def compare_(self, cnn1, cnn2, rnn, input_1, input_2):
        input_1_list = input_tolist(input_1)
        input_2_list = input_tolist(input_2)

        fts_1, fts_2 = [], []
        for input_1 in input_1_list:
            ft_1 = cal_face_tensor(cnn1, input_1)
            fts_1.append(ft_1)

        for input_2 in input_2_list:
            ft_2 = cal_face_tensor(cnn2, input_2)
            fts_2.append(ft_2)

        fts_1 = torch.cat(fts_1)
        fts_2 = torch.cat(fts_2)

        embs_1 = rnn(fts_1)
        embs_2 = rnn(fts_2)

        dist = cal_embs_dist(embs_1, embs_2)
        return dist
    
    def checkgan(self, input_):
        stats = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        tf = tt.Compose([
            tt.CenterCrop(224),
            tt.ToTensor(),
            tt.Normalize(*stats)
            ])
        
        a = []
        input_list = input_tolist(input_)
        for img in input_list:
            img = tf(convert(img)).to(self.device).unsqueeze(0)
            a.append(img)
        
        a = torch.cat(a)
        score = torch.zeros((2, 1)).to(self.device)
        score += self.gan1(a)
        score += self.gan2(a)
        return score
    
    def detect_gender(self, cnn, input_):
        input_ = convert(input_)
        ft = cnn(input_)

        if ft is None:
            return [torch.randn((1, 2))] * 3

        output_ = []
        output_.append(self.senet1(ft.unsqueeze(0)))
        output_.append(self.senet2(ft.unsqueeze(0)))
        output_.append(self.senet3(ft.unsqueeze(0)))
        return output_

    def show_eyes_xy(self, cnn, input, ind):
        input = convert(input)
        _, _, points = cnn.detect(input, True)

        return points[ind][:2].tolist()
