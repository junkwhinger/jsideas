---
layout:     post
title:      "Scala로 만들어본 이름점"
date:       2016-10-27 00:00:00
author:     "Jun"
img: 20161030.jpeg
tags: [scala]
---

## 들어가며
<br>

요새 국운이 융성하려는지 우리 전통문화가 집중조명을 받고 있습니다. 그중에 소위 '이름점'이라는 궁합점이 예능 뿐만 아니라 정치 뉴스에서도 비중있게 다루어져 인기인데요, 최근 공부해보고 있는 Scala로 간단히 코드를 짜서 만들어봤습니다.

<hr>

## Scala
<br>

이번에는 Scala를 사용해서 이름점 코드를 짜봅시다. Python으로는 더 익숙하게 만들 수 있겠지만, Scala는 뭔가 이름도 그렇고 간지가 나기 때문에 배워볼 만한 가치가 있습니다. 조금 익숙해지면 Spark를 사용하는데도 좋다는 것은 보너스입니다. <a href="https://datasciencevademecum.wordpress.com/2016/01/28/6-points-to-compare-python-and-scala-for-data-science-using-apache-spark/">여기에서</a> 데이터분야에서의 Scala와 Python 장단점을 더 자세히 확인할 수 있습니다.

조금 Scala를 만져본 감상은 이렇습니다.

- 알아두고 익숙해지면 Spark를 사용하는데 매우 편리하다
- 생소한 함수형 언어라 그런지 Python으로 프로그래밍을 시작한 사람에게 진입장벽이 꽤 높은 것 같다.
- val / var을 지정하거나 변수의 타입을 꼼꼼하게 설정하고 파악하지 않으면 에러때문에 고생한다.
- class와 object가 마구 등장하고 Trait 같은 어려운 개념들이 있다.
- SBT나 MVN 이런 환경 설정이 Python에 비해 복잡한 것 같다.
- 이해하기 어렵지만 어떤 면에서는 직관적으로 데이터를 처리하기에 용이한 것 같다.

<hr>

## Scala 환경 만들기

Terminal에서 바로 실행할 수 있는 Python과는 달리 Scala는 설치해야 될 것이 좀 많습니다. 맥과 윈도우에서 모두 사용할 수 있는 <a href="https://www.jetbrains.com/idea/">IntelliJ</a>를 사용하면 상대적으로 더 편리하게 작업 환경을 만들 수 있습니다. IntelliJ를 설치하고, Scala & SBT plugin을 설치하면 됩니다.

![IntelliJ](/assets/materials/20161027/intellij.png)

여기서 새로 프로젝트를 만든 후 다음과 같이 SBT를 선택합니다.

![Scala - SBT](/assets/materials/20161027/intellij2.png)

만약 SBT가 보이지 않으면 plugin에서 SBT를 찾아 설치해야 합니다.

<a href="http://www.scala-sbt.org/index.html">SBT</a>는 Scala나 Java 등의 프로젝트를 쉽게 빌드해주는 툴입니다. 

![Project Creation](/assets/materials/20161027/intellij3.png)
여기에 프로젝트 이름을 넣으면 Project Location에 해당 프로젝트 폴더가 생깁니다.

![Project Folder](/assets/materials/20161027/intellij4.png)
폴더에 들어가보면 .idea, project, src, target 폴더가 있고, build.sbt, name_chemistry.imi 파일이 생깁니다. (빌드가 진행되는 과정에서는 일부 파일만 보이게 됩니다.) 여기서 일단 신경쓸 부분은 src와 build.sbt입니다. 

### src
src > main > scala안에 Scala object 파일을 만듭니다. 새로 생성하는 파일 옵션에 없는 경우에는 Scala class 옵션을 선택하고, 새로 생성되는 노트 화면에 Class를 Object로 바꾸면 됩니다.

![Scala Object 만들기](/assets/materials/20161027/scala_object.png)

### build.sbt
여기에는 외부 라이브러리를 등록할 수 있습니다. 이름점을 계산하기 위해서는 한글 문자를 초성 / 중성 / 종성으로 분리해야 하므로, <a href="https://mvnrepository.com/artifact/com.twitter.penguin/korean-text/4.1.4">mvnrepository</a>에서 링크를 찾아 등록해주면 됩니다. 

![mvnrepository](/assets/materials/20161027/twitter_sbt.png)

![build.sbt](/assets/materials/20161027/build_sbt.png)
 
build.sbt 안에 libraryDependencies에 해당 주소 스트링을 추가하고, 프로젝트를 리프레시하면 프로젝트 내에서 추가한 라이브러리를 사용할 수 있게 됩니다.

이후에는 아까 처음에 만들었던 Scala Object 화면 안에다가 Scala로 코드를 쓰기만 하면 됩니다.

<hr>

## 이름점 알고리즘
<br>

이름점은 2명의 이름의 획수를 조합하여 이들간의 궁합(케미)를 측정합니다. 이름점 알고리즘은 다음과 같은 방식으로 구성됩니다.

1. 각각 3글자로 구성된 A와 B의 이름을 준비한다.
2. A의 이름은 A1/A2/A3, B는 B1/B2/B3로 구성된다고 할때
3. A1 / B1 / A2 / B2 / A3 / B3 순으로 글자를 배치한다.
4. 그리고 각 글자의 획수를 계산한다. (여기서는 지역마다 획수 계산 방식에 약간 차이가 있기는 하나, 여기서는 직선이나 원을 1획으로 정한다.)
5. 맨 왼쪽부터 순차적으로 이동하면서 서로 이웃한 획수를 더하고 1의 자리를 기록하고, 맨끝까지 계산한 경우 다음 행으로 이동하여 동일한 연산을 수행한다.
6. 행에서 남은 숫자의 갯수가 2개일때까지 5의 연산을 수행한다.
7. 마지막에 남은 숫자가 2개라면 첫번째 숫자를 10의 자리에, 두번째 숫자를 1의 자리에 넣어 최종 궁합을 계산한다. 
8. 이름궁합은 0에서 99 사이에 위치하며, 높으면 높을수록 두 인물간의 궁합이 좋다.

뉴스에서 비중있게 다뤄진 사례를 활용하면 아래와 같이 이름궁합을 계산할 수 있습니다.

![궁합 90! 엄청 높네요](/assets/materials/20161027/sung_lee.jpg)

<hr>

## Scala Code
<br>

### 한글 초성/중성/종성 획수 계산을 위한 Map 만들기
<br>
Scala에는 Python의 Dictionary와 유사한 Map의 개념이 있습니다. 먼저 stroke_dict라는 Map을 만드는데 그안에는 Character와 Int가 들어갈거라고 정의를 해줍니다. 여기서 var을 쓰는 이유는 이후에 자음과 모음의 획수를 넣어주기 위해 변형이 가능한 Map을 만들기 때문입니다. 변경하지 않는 상수는 val로 지정합니다.

{% highlight scala %}

var stroke_dict:Map[Char, Int] = Map()
stroke_dict += ('ㄱ' -> 2)
stroke_dict += ('ㄴ' -> 2)
stroke_dict += ('ㄷ' -> 3)
stroke_dict += ('ㄹ' -> 5)
stroke_dict += ('ㅁ' -> 4)

{% endhighlight%}

이런식으로 ㅏㅑㅓㅕ같은 모음까지 획수를 페어링해서 넣어줍니다. Map을 만들었으니 이제 이름을 각 문자로 자르고, 문자를 초중종성으로 해체한 다음, 획수를 계산하고, 순서를 재배열해서 획수를 더하는 일련의 과정을 반복하면 궁합을 계산할 수 있게 되는 겁니다.

### 획수 계산기
글자가 넘어왔을때 글자를 해체해서 획수를 계산하는 함수를 만들어 봅니다.

{% highlight scala %}

// stroke_counter는 Char를 받아서 Int를 뱉는다
def stroke_counter (a_char:Char) : Int = {

    // 문자(Char)를 받아서 초중종성으로 해체한다.
    var parsed_hangul = Hangul.decomposeHangul(a_char)

    // 변형이 가능한 리스트 버퍼를 만들고 이 안에는 Char가 들어가게 지정한다.
    var char_list = new ListBuffer[Char]()

    // 한글의 초성과 중성을 빼낸다.
    var h_onset = parsed_hangul.onset
    var h_vowel = parsed_hangul.vowel

    // 빼낸 초성과 중성을 리스트버퍼에 넣는다.
    char_list += h_onset
    char_list += h_vowel

    // 만약 종성이 있는 경우 종성을 빼내서 리스트버퍼에 넣는다.
    if (Hangul.hasCoda(a_char)) {
      var h_coda = parsed_hangul.coda
      char_list += h_coda
    }

    // 빼낸 초중종성 각각에 대해서 앞서 지정한 획수 사전에서 획수를 찾는다.
    val res = char_list.map{ i => stroke_dict.getOrElse(i, 0) }

    // 찾은 획수를 모두 더해 해당 문자열의 최종 획수를 산출한다.
    val total_stroke = res.toList.sum

    // 산출한 최종 획수를 반환한다.
    return total_stroke
}

{% endhighlight%}

여기서 .map이라는 함수가 사용되었는데, pandas의 .map(lambda x: foo(x))와 비슷하게 리스트 내의 값을 돌면서 {}안에서 지정한 함수를 수행합니다. 

### 궁합 계산기

{% highlight scala %}

// Int로 구성된 두개의 리스트를 인자로 받아 Unit 값을 리턴합니다.
// 여기서는 최종값을 출력만 하기에 별도로 출력 타입을 지정하지 않았습니다. 
def chemistry_calculator (res1: List[Int], res2: List[Int]): Unit = {

    // 리스트 버퍼를 만들고, 엇갈리게 각 문자열을 집어넣습니다.
    var num_seq = new ListBuffer[Int]()
    num_seq += res1(0)
    num_seq += res2(0)
    num_seq += res1(1)
    num_seq += res2(1)
    num_seq += res1(2)
    num_seq += res2(2)

    // 만든 리스트 버퍼를 리스트로 변환합니다.
    val final_seq = num_seq.toList

    // 궁합 계산을 위한 내부 함수를 하나 더 만드는데, 
    // 앞서 만든 리스트를 받아서 다시 리스트를 뱉습니다.
    def chemistry_main_function (sequence: List[Int]): List[Int] = {

        // 앞에서부터 2개씩 슬라이딩해가면서 더하고 리스트로 만듭니다.
        var result = sequence.sliding(2).map(_.sum).toList
        
        // 만든 리스트에서 1의 자리만 남기기위해 모듈로디바이드합니다.
        var result2 = result.map( i => i%10)

        // 만약 결과 리스트의 Int 개수가 2개이면 연산을 멈추고
        if (result2.length == 2) {
          return result2
        } else {
          // 그렇지 않다면 다시 리스트를 재귀적으로 처리합니다.
          chemistry_main_function(result2)
        }
    }

    // 이름이 엇갈린 획수 문자열을 처리합니다.
    var final_result_list = chemistry_main_function(final_seq)

    // 최종 결과물의 첫번째 Int를 10의 자리에, 두번째 Int를 1의 자리에 넣어 최종 궁합점수를 계산합니다.
    val final_result = final_result_list(0) * 10 + final_result_list(1)

    // 최종 궁합점수를 출력합니다.
    println(final_result)

}

{% endhighlight%}


### 실행
이 모든 코드는 name_chemistry라는 Object 안에 들어있습니다. 그 아래 부분에 다음과 같은 코드를 넣고 테스트해봅시다.

{% highlight scala %}

// 테스트
val a_name = "성완종".toList
val b_name = "이완구".toList

// 각 이름에 대해서 획수 계산
val a_res = a_name.map(i => stroke_counter(i))
val b_res = b_name.map(i => stroke_counter(i))

// 획수를 사용하여 궁합 출력
chemistry_calculator(a_res, b_res)

{% endhighlight%}

![테스트 성공](/assets/materials/20161027/sung_lee_result.png)

뉴스에서 나왔다시피 스칼라 코드를 빌드해서 실행한 결과 90이 나왔습니다! ㅎㅎ
이제 잘 되었으니 다른 인물들에 대해서도 테스트해봅니다 :)

<hr>

## 궁합 테스트

요새 각 분야에 핫한 인물들에 대해서 이름점 궁합을 확인해봅시다.

### 스포츠
요새는 잠시 주춤하지만 얼마전까지만 해도 우리형의 아성을 넘본 우리흥.
![우리흥 - 우리형](/assets/materials/20161027/ronaldo.jpg)
(출처: 중앙일보)
{% highlight scala %}

val a_name = "손흥민".toList
val b_name = "호날두".toList

val a_res = a_name.map(i => stroke_counter(i))
val b_res = b_name.map(i => stroke_counter(i))

chemistry_calculator(a_res, b_res)

{% endhighlight%}

결과는 92!! 역시 이름점은 과학입니다...

### 연예
방금 본 썰전의 유시민, 전원책 변호사의 케미는 어떨까요?
![유시민 - 전원책](/assets/materials/20161027/sull.jpg)
(출처: kmib.co.kr)
{% highlight scala %}

val a_name = "전원책".toList
val b_name = "유시민".toList

val a_res = a_name.map(i => stroke_counter(i))
val b_res = b_name.map(i => stroke_counter(i))

chemistry_calculator(a_res, b_res)

{% endhighlight%}

결과는 63입니다. 이름을 바꿔서도 넣어봐도 58로 궁합 점수가 우리흥형 커플에 비해 떨어집니다. 아무래도 서로 대립을 하는 구조다보니 궁합이 좀 떨어지지 않나 마 그렇게 생각합니다.

### 정치
마지막으로 요새 가장 핫한 분야, 정치에서 핫한 2명의 인물을 뽑아보겠습니다. 랜덤으로 신문기사를 뽑았더니 박근혜 현직 대통령과 그 지인 최순실님이 눈에 띄네요. 

![박근혜 - 최순실](/assets/materials/20161027/park_choi.jpg)
(출처: 뉴스타파)
{% highlight scala %}

val a_name = "박근혜".toList
val b_name = "최순실".toList

val a_res = a_name.map(i => stroke_counter(i))
val b_res = b_name.map(i => stroke_counter(i))

chemistry_calculator(a_res, b_res)

{% endhighlight%}
대망의 결과는!! 79가 나왔습니다. 아무래도 우리흥형 커플처럼 서로 빙의는 하지는 않지만 그렇다고 썰전 커플처럼 서로 대립하지는 않기때문에 적절한 수준에서 궁합이 나온 것 같네요. 대략 친구사이다..정도로 해석을 하면 적절할 것 같습니다. 

<hr>

## 마치며

이번에는 Scala-SBT 환경에서 이름 궁합점 코드를 작성하고 요새 핫한 인물들간의 궁합을 살펴봤습니다. 좀 더 Scala 공부를 열심히 해서 Jar파일도 말아보고 Spark에서 이리저리 돌려봐야겠습니다! 

<a href="http://jsideas.net/scala/2016/10/30/name_chemistry_refactoring.html">이름점 - Scala Code 수정</a>