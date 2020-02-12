# $$+词 表示需要映射成的词
# =左边的词是纠正词
# =右边的表达式是正则表达式模板
# yes 的意思是不进行词映射
# no 的意思是需要进行词映射

$$兴趣
$进去_yes=.*
$心情_yes=.*

$听见_no=没有听见
$听清_no=没有听清
$信息_no=(可以用|没有)信息
$限制_no=没有限制

$$可以
$所以_yes=.*
$看一下_yes=先看一下|看一下吧
$问一下_yes=我问一下|想问一下

$考虑_no=^考虑$
$而且_no=而且.*?(?!(什么))

$$了解
$而且_no=而且.*什么
$调节_no=怎么调节
$聊解_no=先不聊解
$聊天_no=(不|怎么)聊天

$$愿意
$鸳鸯_no=^鸳鸯$
$越南_no=^越南$