import os
import shutil
import subprocess

def fix_dataset(method):
    dataset_path = "D:\\cstnet-tls 1.3\\"
    comand = "D:\\Wireshark\\mergecap.exe -w D:\\test\\%s.pcap D:\\cstnet-tls 1.3\\%s\\*.pcap"
    for p, d, f in os.walk(dataset_path):
        for label in d:
            if label != "0_merge_datas":
                label_domain = label.split(".")[0]
                print(comand % (label_domain, label))

    return 0


def reverse_dir2file():
    path = "E:\\dataset\\"
    for p, d, f in os.walk(path):
        for file in f:
            shutil.move(p + "\\" + file, path)
    return 0

def dataset_file2dir(file_path):
    for parent, dirs, files in os.walk(file_path):
        for file in files:
            label_name = file.split(".pcap")[0]
            os.mkdir(parent + "\\" + label_name)
            shutil.move(parent + "\\" + file, parent + "\\" + label_name + "\\")
    return 0

def file_2_pcap(source_file, target_file):
    cmd = "D:\\Wireshark\\tshark.exe -F pcap -r %s -w %s"
    command = cmd % (source_file, target_file)
    os.system(command)
    return 0


def clean_pcap(source_file):
    target_file = source_file.replace('.pcap', '_clean.pcap')
    clean_protocols = '"not arp and not dns and not stun and not dhcpv6 and not icmpv6 and not icmp and not dhcp and not llmnr and not nbns and not ntp and not igmp and frame.len > 80"'
    cmd = "D:\\Wireshark\\tshark.exe -F pcap -r %s -Y %s -w %s"
    command = cmd % (source_file, clean_protocols, target_file)
    os.system(command)
    return 0

def statistic_dataset_sample_count(data_path):
    dataset_label = []
    dataset_lengths = []

    tls13_flag = 0

    subdirectories = os.listdir(data_path)

    for subdir in subdirectories:
        subdir_path = os.path.join(data_path, subdir)

        if not os.path.isdir(subdir_path):
            continue

        dataset_label.append(subdir)

        file_list = os.listdir(subdir_path)

        if len(file_list) > 0:
            file_num = len(file_list)
            dataset_lengths.append(file_num)
        else:
            print(f"Ignoring empty directory: {subdir_path}")

    print("label samples: ", dataset_lengths)
    print("labels: ", dataset_label)
    return dataset_lengths, dataset_label


if __name__ == '__main__':
    fix_dataset(['method'])
