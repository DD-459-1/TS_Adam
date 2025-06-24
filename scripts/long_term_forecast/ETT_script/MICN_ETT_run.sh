
# 获取当前脚本的绝对路径（即使你在上级目录运行）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 定义脚本列表
scripts=(
    "MICN_ETTh1.sh"
)

# 定义要测试的种子列表
seeds=(2021 2077)

# 定义要测试的batch_size列表
batch_sizes=(32)

# 依次运行每个脚本，对每个种子值和每个batch_size
for seed in "${seeds[@]}"; do
    for bs in "${batch_sizes[@]}"; do
        echo "========== 使用种子: $seed, batch_size: $bs =========="

        for script in "${scripts[@]}"; do
            echo "正在运行: $script (种子: $seed, batch_size: $bs)"
            # 使用绝对路径运行子脚本，并传递种子和batch_size参数
            bash "${SCRIPT_DIR}/${script}" "$seed" "$bs" || {
                echo "运行 $script 时出错"
                exit 1
            }
            echo "$script (种子: $seed, batch_size: $bs) 运行完成"
            echo "-------------------------"
        done
    done
done

echo "所有脚本已成功运行完毕"