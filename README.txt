# settings in main.py

TEST = Ture
# when TEST = Ture, the code just test the trained Multi-VAE model
# when TEST = False, the code will train Multi-VAE model 

# run the code：
python main.py

# visualize the generative model：
python Load_model_visual.py

# paper website
https://openaccess.thecvf.com/content/ICCV2021/html/Xu_Multi-VAE_Learning_Disentangled_View-Common_and_View-Peculiar_Visual_Representations_for_Multi-View_ICCV_2021_paper.html

# citation
@InProceedings{Xu_2021_ICCV,
    author    = {Xu, Jie and Ren, Yazhou and Tang, Huayi and Pu, Xiaorong and Zhu, Xiaofeng and Zeng, Ming and He, Lifang},
    title     = {Multi-{VAE}: Learning Disentangled View-Common and View-Peculiar Visual Representations for Multi-View Clustering},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2021},
    pages     = {9234-9243}
}
