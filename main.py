import handler as h

def main():
    mnist_model, vid = h.init("./docs/model.zip", 0)
    count = 0
    while (True):
        # Obtain the current frame and image
        frame, image = h.get_image(vid)
        #save frame as a png
        h.cv2.imwrite("./before_preprocessing/frame%d.png" % count, frame)
    
        # Image prepocessing
        image = h.transform_frame(image)

        # Image evaluation
        salida = mnist_model(image)
        # Show real time visual results
        transformed = h.visual_testing(salida, image, frame, count)
        #save image as a png
        h.cv2.imwrite("./after_preprocessing/image%d.png" % count, transformed)
        count += 1

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if h.cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    h.cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
