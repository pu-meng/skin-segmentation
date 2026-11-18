# CPUBone: Efficient Vision Backbone Design for Devices with Low Parallelization Capabilities
# Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni
# Conference on Computer Vision and Pattern Recognition (CVPR), 2025

import argparse, time
import math
import os, sys
import numpy as np
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import torch  

# import multiprocessing
from termcolor import colored


def mac_it(model, imgsize, chans=3, ret_macs=False):
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (chans, imgsize, imgsize), as_strings=True,
                                print_per_layer_stat=False, verbose=False)
    printstring = '{:<30}  {:<8}'.format('Computational complexity: ', macs) +" | params: "+ str(params)+ " | imgsize: "+ str(imgsize)
    print(printstring)
    if ret_macs:
        return macs
    return printstring
# from merge_conv_bn import fuse


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[torch.Tensor]:
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

from torch.profiler import profile, record_function, ProfilerActivity
def profile_it(args, model):
    args.gpu = 0
    inputs = torch.randn(5, 3, 224, 224).to("cuda:%d" % args.gpu)
    model = model.to("cuda:%d" %args.gpu)
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=False, with_stack=True) as prof: # , ProfilerActivity.CUDA
        with record_function("model_inference"):
            model(inputs)
    
    # print(vars(prof.key_averages(group_by_input_shape=True)))
    # print(dir(prof.key_averages(group_by_input_shape=True)))
    # print(repr(prof.key_averages(group_by_input_shape=True)))
    # return
    
    # res = prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=50)
    res = prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=50)
    res = prof.key_averages(group_by_stack_n=5)
    
    print(res)
    
    with open("temp_data/profiler.txt", "w") as write_file:
        write_file.write(str(res))


def benchmark_it(args, model, outfile="", bsize=200, iterations=100, cpu=False):
    if args.bench:
        from deepspeed.profiling.flops_profiler import get_model_profile
        from deepspeed.accelerator import get_accelerator
    if cpu:
        model = model.to("cpu")
    else:
        model = model.to("cuda:0")
    
    if len(outfile)<1:
        outfile = "temp_data/"+args.model+"_bs"+str(bsize)+("_cpu" if cpu else "")+"_sz"+str(args.image_size)+".txt"
                
    if args.bench:
        with get_accelerator().device(0):
            print("num devices:",torch.cuda.device_count(), torch.cuda.get_device_name(0))

            
            flops, macs, params, flopmodel = get_model_profile(model=model, # model
                                        input_shape=(bsize, 3, args.image_size, args.image_size), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                        args=None, # list of positional arguments to the model.
                                        kwargs=None, # dictionary of keyword arguments to the model.
                                        print_profile=True, # prints the model graph with the measured profile attached to each module
                                        detailed=True, # print the detailed profile
                                        iterations=iterations,
                                        module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                        top_modules=1, # the number of top modules to print aggregated profile
                                        warm_up=15, # the number of warm-ups before measuring the time of each module
                                        as_string=False, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                        output_file=outfile, # path to the output file. If None, the profiler prints to stdout.
                                        ignore_modules=None) # the list of modules to ignore in the profiling
            print(macs)
            print("Total Macs:","%f MMACS" % (macs/(iterations*bsize*1_000_000)))
            # print(flopmodel)
    if args.bench or args.testrun:
        tim_res = testrun_it_old(model=model, image_size=args.image_size, batch_size=bsize, cpu=cpu, iterations=iterations, optit=args.optit, args=args)
        with open(outfile,"a") as write_file:
            write_file.write("\n\n\n----------------------\n"+str(tim_res)+ " ms")


def other_macit(args, model):
    from deepspeed.profiling.flops_profiler import get_model_profile
    from deepspeed.accelerator import get_accelerator
    # with get_accelerator().device(0):  
    flops, macs, params = get_model_profile(model=model, # model
                                input_shape=(1, 3, args.image_size, args.image_size), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                args=None, # list of positional arguments to the model.
                                kwargs=None, # dictionary of keyword arguments to the model.
                                print_profile=True, # prints the model graph with the measured profile attached to each module
                                detailed=True, # print the detailed profile
                                # iterations=1,
                                module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                top_modules=1, # the number of top modules to print aggregated profile
                                warm_up=1, # the number of warm-ups before measuring the time of each module
                                as_string=False, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                output_file="b1_grouping.txt", # path to the output file. If None, the profiler prints to stdout.
                                ignore_modules=None) # the list of modules to ignore in the profiling

    print(colored("Total Macs:","light_green"),"%f MMACS" % (macs/(1_000_000)))
        

def testrun_it_old(model, image_size, proc=-1, iterations=100, batch_size=300, cpu=False, return_dict=None, optit=False, args=None):
    device = "cpu" if cpu else "cuda:0"
    inp = torch.randn(batch_size, 3, image_size, image_size).to(device)
    if optit:
        model.eval()
        model = torch.jit.script(model)#, example_inputs=[inp])    
        model = torch.jit.optimize_for_inference(model)
    if args.quantit:
        backend = "qnnpack" #"qnnpack"
        model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend
        model_static_quantized = torch.ao.quantization.prepare(model, inplace=False)
        model_static_quantized = torch.ao.quantization.convert(model_static_quantized, inplace=False)
        model = model_static_quantized
    model.eval()
    model.to(device)


    with torch.inference_mode():
        for i in range(5):
            out = model(inp)
        
        timings = []
        for i in range(1):
            inp = torch.randn(batch_size,3,image_size,image_size).to(device)
            model.eval()
            model.to(device)
            for i in range(10):
                out = model(inp)
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # print(inp.shape)
            for i in range(iterations):
                if cpu:
                    start_time = time.time()
                else:
                    starter.record()
                out = model(inp)
                if cpu:
                    timings.append(time.time() - start_time)
                else:
                    ender.record()
                    torch.cuda.synchronize()
                    timings.append(starter.elapsed_time(ender)/inp.shape[0])
                # print(timings[-1])
            model.to("cpu")
        
    timings = np.array(timings)
    print("backbonetime:", timings[-1],"| median: %.3f" % (np.median(timings)), "mean: %.3f" % (np.mean(timings)), "Throughput: %.3f" % (1000/np.median(timings)))
    
    if not return_dict is None:
        return_dict[proc] = {"median": np.median(timings), "mean": np.mean(timings)}
    else:
        return np.median(timings)


def testrun_it(proc, args, return_dict=None):
    device = "cpu" if args.cpu else "cuda:0"
    inp = torch.randn((args.batch_size, 3, args.image_size, args.image_size)).to(device).float()

    fac = args.facbs
    inp = torch.randn((args.batch_size*fac**2, 3, args.image_size//fac, args.image_size//fac)).to(device).float()
    
    # inp = torch.ones(args.batch_size, 3, args.image_size, args.image_size).to(device)
    
    # return_dict[proc] = inp.shape
    if args.ds:
        model = torch.jit.load(os.path.join("Wtotal_model",args.ds+".pt"))
        modelname = args.ds
    else:
        model = get_model(args.model, args.other,image_size=args.image_size, args=args).to(device)
        modelname = args.model if len(args.other)<1 else args.other

    if args.optit:
        model.eval()
        model = torch.jit.script(model)#, example_inputs=[inp])
        model = torch.jit.optimize_for_inference(model)

    if args.ds:
        args.image_size = 512

    ############################################
    model.eval()
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for i in range(10):
            _ = model(inp)
        
        # print(inp.shape)
        # print(inp.get_device())
        timings = []
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
        if True:
            for i in range(args.iterations):
                if args.cpu:
                    start_time = time.time()
                else:
                    starter.record()
                
                for i in range(args.multexec):
                    out = model(inp)
                    # print([out[key].shape for key in out])
                if args.cpu:
                    timings.append(1000*(time.time()-start_time)/(args.multexec*args.batch_size))
                else:
                    ender.record()
                    torch.cuda.synchronize()
                    timings.append(starter.elapsed_time(ender)/(args.multexec * args.batch_size))
                # print(timings[-1])
                
            timings = np.array(timings)
            med = np.median(timings)
            mean = np.mean(timings)
            return_dict[proc] = {"median": med, "mean": mean}
        
        else:
            starter.record()
            for i in range(args.iterations):
                out = model(inp)
                
            ender.record()
            torch.cuda.synchronize()
            med = starter.elapsed_time(ender)/(inp.shape[0]*args.iterations)
            mean = med
            return_dict[proc] = {"median": med, "mean": mean}


def get_model(modelname, other="", weight_url="", image_size=224, less_layers=0, args=None):
    if len(other) > 0:
        from other_models.shvit_temp import shvit_s1, shvit_s2, shvit_s3, shvit_s4
        from other_models.efficientModulation import efficientmod
        from other_models.mobilevig import get_mobile_vig
        
        if "mambavision" in other:
            # from transformers import AutoModelForImageClassification
            # model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-"+other.replace("mambavision","")+"-1K", trust_remote_code=True)
            from other_models.mambavision import get_model
            model = get_model(other.replace("mambavision",""))
        elif "efficientmod" in other:
            model = efficientmod(other.replace("efficientmod","")) 
        elif "mobilevig" in other:
            model = get_mobile_vig(other.replace("mobilevig",""))
        elif "iformer" in other:
            from other_models.inceptionFormer import get_iformer
            model = get_iformer(other.replace("iformer",""))
        elif "mobileone" in other: 
            from other_models.mobileone_repr import get_mobileone
            model = get_mobileone(other.replace("mobileone",""))
        elif "ffnet" in other:
            from other_models.ffnet import get_ffnet
            model = get_ffnet(other.replace("ffnet",""))
        elif "fasternet" in other:
            from other_models.fasternet import get_fasternet
            model = get_fasternet(other.replace("fasternet",""))
        elif "ghostnet" in other:
            from other_models.ghostnetv2 import get_ghostnet
            model = get_ghostnet(other.replace("ghostnet",""))
        elif "efficientvit" in other:
            from other_models.efficientvitmemory import get_model_effvit
            model = get_model_effvit(other.replace("efficientvit",""), image_size=image_size)
        elif "edgevit" in other:
            from other_models.edgevit import edegvit_model
            model = edegvit_model(other.replace("edgevit",""), image_size)
        elif "pvtv2" in other:
            from other_models.pvtv2 import pvtv2_model
            model = pvtv2_model(other.replace("pvtv2",""))
        elif "FAT" in other:
            from other_models.FAT import get_fat_model
            model = get_fat_model(other.replace("FAT",""))
        elif "mobilenetv3" in other:
            from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
            model = eval("mobilenet_v3_"+other.replace("mobilenetv3","")+"()")
        elif "coatnet" in other:
            from other_models.coatnet import get_coatnet
            model = get_coatnet(other.replace("coatnet",""))
        elif "efficientformerv2" in other:
            from other_models.efficientformerv2 import get_efficientformer
            model = get_efficientformer(other.replace("efficientformerv2",""), image_size)
        elif "resnet" in other:
            from torchvision.models import resnet50, resnet18, resnet101
            model = eval(other+ "()")
        else:
            import importlib
            repvit = importlib.import_module("other_models.repvit")
            model = getattr(repvit, other)()
        # model = other
        # print(model)
    else:
        from cpubone_model import get_cpubone
        model = get_cpubone(config_path="configs/cls/imagenet/"+modelname+".yaml", checkpoint_path=".exp/cls/imagenet/"+modelname+"/checkpoint/checkpoint.pt", pretrained=not args.testrun)
        
    return model 

def run_total(model, device, args, proc_num=0, return_dict=None):
    
    if args.optit:
        model.eval()
        model = model.to(device)
        model = torch.jit.script(model)#, example_inputs=[inp])    
        model = torch.jit.optimize_for_inference(model)
    model.eval()
    
    
    inp = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(device)
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
    with torch.inference_mode():
        for i in range(10):
            out = model(inp)
        
        if not args.cpu:
            torch.cuda.synchronize()
        start_time = time.time()

        for i in range(args.iterations):
            for i in range(args.multexec):
                out = model(inp)  #[N,1000]
            # torch.cuda.synchronize()    
            inp = inp * out[0,0]
            
        if not args.cpu:
            torch.cuda.synchronize()
        passed_time = (time.time() - start_time) * (1000/(args.iterations * args.batch_size * args.multexec))
    
    if not return_dict is None:
        return_dict[proc_num] = passed_time 
    
    return passed_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("savedir", metavar="FILE", nargs="?", default=None, help="config file (alternative to --model)")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--testrun", action="store_true", default=False)
    parser.add_argument("--latency", action="store_true", default=False)
    parser.add_argument("--optit", action="store_true", default=False)
    parser.add_argument("--ds", type=str, default="")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--other", type=str, default="")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--batch_size", help="batch size per gpu", type=int, default=200)
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("-j", "--workers", help="number of workers", type=int, default=10)
    parser.add_argument("--crop_ratio", type=float, default=0.95)
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--prof", action="store_true", default=False)
    parser.add_argument("--bench", action="store_true", default=False)
    parser.add_argument("--fusebn", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--maconly", action="store_true", default=False)
    parser.add_argument("--quantit", action="store_true", default=False)
    parser.add_argument("--nomac", action="store_true", default=False)
    parser.add_argument("--total_exec", action="store_true", default=False)
    parser.add_argument("--multexec", type=int, default=1)
    parser.add_argument("--facbs", type=int, default=1)
    parser.add_argument("--chaninp", type=int, default=3)
    parser.add_argument("--torch_conversion", action="store_true", default=False)
    
    
    args = parser.parse_args()
    # torch.backends.cudnn.benchmark=True
    # if args.gpu == "all":
    #     device_list = range(torch.cuda.device_count())
    #     args.gpu = ",".join(str(_) for _ in device_list)
    # else:
    #     device_list = [int(_) for _ in args.gpu.split(",")]
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    args.batch_size = args.batch_size #* max(len(device_list), 1)

    if args.savedir is None:
        if args.model is None:
            parser.error("provide a model name via positional argument or --model")
        args.savedir = args.model
    args.model = args.savedir
    args.savedir = ".exp/cls/imagenet/" + args.savedir
    args.weight_url = args.savedir+"/checkpoint/checkpoint.pt"
    
    # args.weight_url = args.savedir+"/checkpoint/model_best.pt" # TODO REMOVE
    
    
    # try:
    #     config = setup.setup_exp_config(args.savedir+"/config.yaml", recursive=True, opt_args=None)
    # except:
    
    if not args.testrun:
        data_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                args.path,
                transforms.Compose(
                    [
                        transforms.Resize(
                            int(math.ceil(args.image_size / args.crop_ratio)), interpolation=InterpolationMode.BICUBIC
                        ),
                        transforms.CenterCrop(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )

    model = get_model(args.model, args.other, args.weight_url, args.image_size, args=args)

    device = "cpu" if args.cpu else "cuda:0"
    model.eval()
    model(torch.randn(1,3,args.image_size, args.image_size))
    if args.fusebn:
        model = fuse(model)
    # with open("temp_data/modelprint.txt", "w") as write_file:
    #     write_file.write(str(model))
    # with open("modelprint.txt", "w") as write_file:
    #     write_file.write(str(model))
    

    if not args.nomac:
        mac_it(model=model,imgsize=args.image_size)
        other_macit(args, model)
    if args.maconly:
        return
    
    if args.torch_conversion:
        os.makedirs("Wtorchmodels", exist_ok=True)

        try:
            scripted_it = True
            with torch.no_grad():
                model = torch.jit.script(model.cuda(), example_inputs=[(1, args.chaninp, args.image_size, args.image_size)])#, example_inputs=[inp])    
        except:
            scripted_it = False
            print("Torchscript conversion failed! Uses jit_trace now!")
            model = torch.jit.trace(model.cuda(), example_inputs=torch.randn(1, args.chaninp, args.image_size, args.image_size).cuda())#, example_inputs=[inp])    
            
        # model = torch.jit.optimize_for_inference(model)
        modelname = args.model if len(args.other)<1 else args.other
        torchpath = os.path.join("Wtorchmodels",modelname+"_"+str(args.image_size)+(".tscript" if scripted_it else ".ttraced"))
        torch.jit.save(model,torchpath)
        print("jit conversion and saving successful!")
        return

    if args.total_exec:
        if args.jobs > 1:
            if True: # GOOD WAY
                import torch.multiprocessing as mp
                mp.set_start_method("spawn",force=True)
                manager = mp.Manager()
                return_dict = manager.dict()
                jobs = []
                for i in range(args.jobs): # model, args, proc=-1, iterations=100, batch_size=300, cpu=False
                    p = mp.Process(target=run_total, args=(model, device, args, i,return_dict))#, args=(model, args.image_size, i, args.iterations, args.batch_size, args.cpu))
                    jobs.append(p)
                    p.start()
                    # print("Started",p.name)
                    
                for proc in jobs:
                    proc.join()
                meantime = np.mean([return_dict[key] for key in return_dict])
                print(colored("Mean time needed:","light_green"), meantime) # colored(args.model,"light_green")+" on ImageNet")
                return
            
            else:
                import multiprocessing as mp
                mp.set_start_method("spawn",force=True)
                manager = mp.Manager()
                return_dict = manager.dict()
                jobs = []
                for i in range(args.jobs): # model, args, proc=-1, iterations=100, batch_size=300, cpu=False
                    p = mp.Process(target=run_total, args=(model, device, args, i,return_dict))#, args=(model, args.image_size, i, args.iterations, args.batch_size, args.cpu))
                    jobs.append(p)
                    p.start()
                    # print("Started",p.name)
                    
                for proc in jobs:
                    proc.join()
                meantime = np.mean([return_dict[key] for key in return_dict])
                print(colored("Mean time needed:","light_green"), meantime) # colored(args.model,"light_green")+" on ImageNet")
                return
            
        passed_time = run_total(model, device, args)
        print("Time needed:", passed_time, "bs:", args.batch_size )
        return
    if args.latency: # model, args, iterations=100, batch_size=300, cpu=False

        if args.jobs > 1:
            import multiprocessing as mp
            mp.set_start_method("spawn",force=True)
            
            # args.batch_size = 1
            manager = mp.Manager()
            return_dict = manager.dict()
            jobs = []  
            for i in range(args.jobs ): # model, args, proc=-1, iterations=100, batch_size=300, cpu=False
                p = mp.Process(target=testrun_it, args=(i, args, return_dict))#, args=(model, args.image_size, i, args.iterations, args.batch_size, args.cpu))
                jobs.append(p)
                p.start()
                # print("Started",p.name)
                
            for proc in jobs:
                proc.join()
            # print("return dict:",return_dict)
            meanmed = np.mean([return_dict[key]["median"] for key in return_dict])
            meanmean = np.mean([return_dict[key]["mean"] for key in return_dict])
            print("Meanmed:", meanmed, "Meanmean:", meanmean, [return_dict[key]["median"] for key in return_dict])
        else:
            ret_dict = {}
            testrun_it(0,args,ret_dict)
            print("Med:",{key:float(ret_dict[0][key]) for key in ret_dict[0]})
        return
        # testrun_it(procnum=procnum, model=model, args=args, iterations=args.iterations, batch_size=args.batch_size, cpu=args.cpu)
    if args.prof:
        profile_it(args, model)
        return 
    if args.bench or args.testrun:
        benchmark_it(args, model, bsize=args.batch_size, cpu=args.cpu, iterations=args.iterations)
        return
    
    ## reset BN TODO
    if False:
        from cpubone.models.nn.norm import reset_bn
        for i in data_loader:
            print(i[0].shape,i[1].shape)
            print(i.shape)
        ###CONTINUE!!!!
        reset_bn(model,data_loader)

    
    model = torch.nn.DataParallel(model).cuda()
    model.eval()


    # Timing
    total_time_list = []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    top1_sum = top5_sum = n = 0
    with torch.inference_mode():
        with tqdm(total=len(data_loader), desc="Eval "+ colored(args.model,"light_green")+" on ImageNet") as t:
            for images, labels in data_loader:
                images, labels = images.cuda(), labels.cuda()

                starter.record()
                output = model(images)
                bsize = images.shape[0]
                ender.record()
                torch.cuda.synchronize()
                total_time_list.append(starter.elapsed_time(ender)/images.shape[0])

                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top1_sum += acc1[0].item() * bsize
                top5_sum += acc5[0].item() * bsize
                n += bsize
                t.set_postfix(
                    {
                        "top1": f"{top1_sum/n:.2f}",
                        "medtim:": np.median(np.array(total_time_list)),
                        "top5": f"{top5_sum/n:.2f}",
                        "resolution": images.shape[-1],
                    }
                )
                t.update(1)
    print("len total timelist:",len(total_time_list),"bsize:",bsize)
    print("median time needed:",np.median(np.array(total_time_list))," mean:",np.mean(np.array(total_time_list)))
    print(f"Top1 Acc={top1_sum/n:.3f}, Top5 Acc={top5_sum/n:.3f}")


if __name__ == "__main__":
    main()

#  python eval_cls_model.py b1_testing10 --gpu 0  --batch_size 20 --image_size 224 --testrun --other custom1 --cpu
