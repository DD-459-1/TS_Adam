
# 获取当前脚本的绝对路径（即使你在上级目录运行）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 定义脚本列表
scripts=(
    "PatchTST_ETTh1.sh"
    "PatchTST_ETTh2.sh"
    "PatchTST_ETTm1.sh"
    "PatchTST_ETTm2.sh"
    "SegRNN_ETTh1.sh"
    "SegRNN_ETTh2.sh"
    "SegRNN_ETTm1.sh"
    "SegRNN_ETTm2.sh"


)


# 定义要测试的种子列表
seeds=(123 2077 2021)

# 依次运行每个脚本，对每个种子值
for seed in "${seeds[@]}"; do
    echo "========== 使用种子: $seed =========="

    for script in "${scripts[@]}"; do
        echo "正在运行: $script (种子: $seed)"
        # 使用绝对路径运行子脚本，并传递种子参数
        bash "${SCRIPT_DIR}/${script}" "$seed" || {
            echo "运行 $script 时出错"
            exit 1
        }
        echo "$script (种子: $seed) 运行完成"
        echo "-------------------------"
    done
done

echo "所有脚本已成功运行完毕"