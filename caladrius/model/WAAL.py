#Todo: all irrelevant
import torch
import copy

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import grad
from model.networks.inception_siamese_network import InceptionSiameseNetwork

def caladrius_waal_extractor(qsn):
    '''
    Splits the Caladrius model into feature extractor, classifier and discriminator
    :param qsn: Caladrius model
    :returns qsn: The adjusted model
    '''

     #Put in this function so no errors are given through the create_logger function (an error is given that this already exists)

    qsn.net_fea = FeatureExtractorQsn(qsn.model)
    qsn.net_clf = ClassifierQsn(qsn.model)
    qsn.net_dis = DiscriminatorQsn(qsn.model)
    # qsn.net_dis = Net1_dis_WAAL_type() #Can be used to select another discriminator architecture
    return qsn

class FeatureExtractorQsn(InceptionSiameseNetwork):
    '''
    Creates the feature extraction model based in InceptionV3, as separate class such that forward method can be changed
    '''
    def __init__(self, qsn_model):
        super().__init__()
        self.left_network = copy.deepcopy(qsn_model.left_network)
        self.right_network = copy.deepcopy(qsn_model.right_network)
        self.similarity = torch.nn.Identity()
        self.output = torch.nn.Identity()

    def forward(self, image_1, image_2):
        """
        Define the feedforward sequence
        Args:
            image_1: Image fed in to left network
            image_2: Image fed in to right network

        Returns:
            extracted features
        """

        left_features = self.left_network(image_1)
        right_features = self.right_network(image_2)


        if self.training:
            left_features = left_features[0]
            right_features = right_features[0]

        features = torch.cat([left_features, right_features], 1)

        return features

class ClassifierQsn(InceptionSiameseNetwork):
    '''
    Creates the classifier, as separate class such that forward method can be changed
    '''
    def __init__(self, qsn_model):
        super().__init__()
        self.left_network = torch.nn.Identity()
        self.right_network = torch.nn.Identity()
        self.similarity = copy.deepcopy(qsn_model.similarity)
        self.output = copy.deepcopy(qsn_model.output)

    def forward(self, features):
        """
        Define the feedforward sequence
        Args:
            features: output of FeatureExtractorQsn

        Returns:
            Predicted output
        """

        sim_features = self.similarity(features)

        output = self.output(sim_features)

        return output


class DiscriminatorQsn(InceptionSiameseNetwork):
    '''
    Creates the classifier, as separate class such that forward method can be changed
    '''
    def __init__(self, qsn_model):
        super().__init__()
        self.left_network = torch.nn.Identity()
        self.right_network = torch.nn.Identity()
        self.similarity = copy.deepcopy(qsn_model.similarity)
        self.output = torch.nn.Linear(in_features=512, out_features=1, bias=True)

    def forward(self, features):
        """
        Define the feedforward sequence
        Args:
            features: output of FeatureExtractorQsn

        Returns:
            Predicted output
        """

        sim_features = self.similarity(features)

        output = self.output(sim_features)

        #Sigmoid is used to make it 0 or 1
        output = torch.sigmoid(output)

        return output

class Net1_dis_WAAL_type(torch.nn.Module):

    """
    Discriminator network, output with [0,1] (sigmoid function)

    """
    # def __init__(self, qsn_model):
    #     super().__init__()
    #     self.left_network = torch.nn.Identity()
    #     self.right_network = torch.nn.Identity()
    #     self.similarity = copy.deepcopy(qsn_model.similarity)
    #     self.output = torch.nn.Linear(in_features=512, out_features=1, bias=True)
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1024, 50)
        self.fc2 = torch.nn.Linear(50, 1)

    def forward(self,x):
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


def set_requires_grad(model, requires_grad=True):
    """
    Used in training adversarial approach, code from https://github.com/cjshui/WAAL
    :param model:
    :param requires_grad:
    :return:
    """

    for param in model.parameters():
        param.requires_grad = requires_grad

def gradient_penalty(critic, h_s, h_t):
    ''' Gradeitnt penalty approach, code from https://github.com/cjshui/WAAL'''
    if h_s.is_cuda: #If statement added to make it run as well when on cpu
        alpha = torch.rand(h_s.size(0), 1).cuda()
    else:
        alpha = torch.rand(h_s.size(0), 1)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
    # interpolates.requires_grad_()
    preds = critic(interpolates) #Added note polle: critic is the discriminator network
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()

    return gradient_penalty


def WAAL_query(qsn, query_num, pool_loader):

    """
    adversarial query strategy WAAL, from https://github.com/cjshui/WAAL

    :param query_num: number of images to be selected
    :return:

    """

    # idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

    # prediction output probability
    probs = predict_prob(qsn, pool_loader)



    # uncertainly score (three options, single_worst, L2_upper, L1_upper)
    # uncertainly_score = self.single_worst(probs)
    uncertainly_score = 0.5* single_worst(probs) + 0.5* L1_upper(probs)

    # print(uncertainly_score)


    # prediction output discriminative score
    dis_score = pred_dis_score(qsn, pool_loader)

    # print(dis_score)

    print('------mean and std for uncertainty and dis---------')
    print(uncertainly_score[torch.isfinite(uncertainly_score)].mean()) #approx 7.7 for caladrius, 17.5 for waal value -> seems 4/10 so class difference, only on average laregr values when only 4 classes as well
    # Further, dis_score is comparable to WAAL implementation. Hence, to have the same distribution between uncertainty
    # and dis_score, selection=4 could be used. However, to check somewhat more, the mean and std of both are given
    # over iterations. Selection meanwhile is set to 10
    print(uncertainly_score[torch.isfinite(uncertainly_score)].std())
    print('dis scores unlabelled')
    print(dis_score[torch.isfinite(dis_score)].mean())
    print(dis_score[torch.isfinite(dis_score)].std())
    print(dis_score.min())
    print(dis_score.max())
    # print(dis_score.numpy())
    print('---------------------------------------------------------------')

    selection = 5
    # computing the decision score
    total_score = uncertainly_score - selection * dis_score
    # print(total_score)

    # sort the score with minimal query_number examples
    # expected value outputs from smaller to large
    b = total_score.sort()[1][:query_num]
    # print(total_score[b])

    return b.numpy()

def predict_prob(qsn, loader):

    """
    prediction output score probability
    :param X:
    :param Y: NEVER USE the Y information for direct prediction
    :return:
    """

    qsn.net_fea.eval()
    qsn.net_clf.eval()

    probs = torch.zeros([len(loader.dataset), qsn.number_classes])
    with torch.no_grad():

        for idx, (filename, image1, image2, labels) in enumerate(loader, 1):
            #first create indices of images checked at this moment
            idxs = np.arange((idx-1)*len(labels),idx*len(labels))

            image1 = image1.to(qsn.device)
            image2 = image2.to(qsn.device)

            latent = qsn.net_fea(image1,image2) #this part could be directly estimated using qsn.model, as clf and fea are pasted in there already
            out    = qsn.net_clf(latent)
            prob = F.softmax(out, dim=1)
            probs[idxs] = prob.cpu()
    return probs

def pred_dis_score(qsn,loader):

    """
    prediction discrimnator score
    :param X:
    :param Y:  FOR numerical simplification, NEVER USE Y information for prediction
    :return:

    """
    qsn.net_fea.eval()
    qsn.net_dis.eval()

    scores = torch.zeros(len(loader.dataset))

    with torch.no_grad():
        for idx, (filename, image1, image2, labels) in enumerate(loader, 1):
            # first create indices of images checked at this moment
            idxs = np.arange((idx - 1) * len(labels), idx * len(labels))

            image1 = image1.to(qsn.device)
            image2 = image2.to(qsn.device)

            latent = qsn.net_fea(image1,image2)
            out = qsn.net_dis(latent).cpu()
            scores[idxs] = out.view(-1) #Check further, same as predict_prob above this funct
    return scores

def L2_upper(probas):

    """
    Return the /|-log(proba)/|_2

    :param probas:
    :return:  # unlabeled \times 1 (float tensor)

    """

    value = torch.norm(torch.log(probas),dim=1)

    return value


def L1_upper(probas):

    """
    Return the /|-log(proba)/|_1
    :param probas:
    :return:  # unlabeled \times 1

    """
    value = torch.sum(-1*torch.log(probas),dim=1)

    return value


def single_worst(probas):

    """
    The single worst will return the max_{k} -log(proba[k]) for each sample

    :param probas:
    :return:  # unlabeled \times 1 (tensor float)

    """

    value,_ = torch.max(-1*torch.log(probas),1)

    return value