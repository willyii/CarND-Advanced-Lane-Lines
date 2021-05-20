from image_pipeline import image_pipeline
import matplotlib.pyplot as plt

if __name__ == '__main__':
    test_img = plt.imread("../test_images/test3.jpg")

    ans = image_pipeline(test_img)
    plt.imsave("ans.jpg", ans, cmap ='gray')

    


