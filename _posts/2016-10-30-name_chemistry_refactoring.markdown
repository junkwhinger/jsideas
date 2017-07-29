---
layout:     post
title:      "이름점 - Scala Code 수정"
date:       2016-10-30 00:00:00
author:     "Jun"
categories: "Scala"
image: /assets/name_chemistry/header.jpg
---

## Scala로 만들어 본 이름점
<br>

며칠 전 <a href="http://jsideas.net/scala/2016/10/27/name_chemistry.html">Scala를 사용해서 간단한 이름점</a>을 만들어봤다. 일단 오류 없이 돌아가는 코드를 만들기는 했지만 다음날보니 어색한 문장과 습관이 눈에 들어왔다. 외형은 Scala로 쓰기는 했지만 var/val을 떼어놓고 보면 Python이나 R코드처럼 보인달까. 뭔가 어색한 부분이 많았다. 주변의 도움과 구글링을 통해 수정한 부분을 기록으로 남긴다.
<hr>

## 초중종성 사전 -> 함수
오리지널 버전에서는 변형가능한 Map을 만들고 거기에 문자열과 획수 페어를 하나씩 추가했다. 그리고 뒤에서 만든 초중종성에 해당하는 획수를 추출하는 식으로 만들었었다.

{% highlight scala %}

var stroke_dict:Map[Char, Int] = Map()
stroke_dict += ('ㄱ' -> 2)
stroke_dict += ('ㄴ' -> 2)
stroke_dict += ('ㄷ' -> 3)
stroke_dict += ('ㄹ' -> 5)
stroke_dict += ('ㅁ' -> 4)

// ...

// 빼낸 초중종성 각각에 대해서 앞서 지정한 획수 사전에서 획수를 찾는다.
val res = char_list.map{ i => stroke_dict.getOrElse(i, 0) }
{% endhighlight%}

이렇게 작업한 코드를 하나로 합쳐서 함수화시켜봤다.


{% highlight scala %}

def stroke_finder(a_char: Char): Int = {
      val res = a_char match {
        case 'ㅇ' => 1
        case 'ㄱ' | 'ㄴ' | 'ㅅ' => 2
        case 'ㄷ' | 'ㅈ' | 'ㅋ' | 'ㅎ' => 3
        case 'ㅁ' | 'ㅂ' | 'ㅊ' | 'ㅌ' | 'ㅍ' | 'ㄲ' | 'ㅆ' => 4
        case 'ㄹ' => 5
        case 'ㄸ' | 'ㅉ' => 6
        case 'ㅃ' => 8

        case 'ㅡ' | 'ㅣ' => 1
        case 'ㅏ' | 'ㅓ' | 'ㅗ' | 'ㅜ' | 'ㅢ' => 2
        case 'ㅑ' | 'ㅕ' | 'ㅛ' | 'ㅠ' | 'ㅐ' | 'ㅔ' | 'ㅚ' | 'ㅟ' => 3
        case 'ㅒ' | 'ㅖ' | 'ㅘ' | 'ㅝ' => 4
        case 'ㅙ' | 'ㅞ' => 5

        case 'ㄳ' => 4
        case 'ㄵ' | 'ㄶ' => 5
        case 'ㅄ' => 6
        case 'ㄺ' | 'ㄽ' => 7
        case 'ㅀ' => 8
        case 'ㄻ' | 'ㄼ' | 'ㄾ' | 'ㄿ' => 9

        case _ => 0
      }
      return res
    }

// ...

val total_stroke = parsed_list.map{ i => stroke_finder(i) }.sum

{% endhighlight%}
stroke_finder라는 함수를 만들고, 한글 문자열 하나를 받아 사전에서 찾은 후 횟수를 반환하도록 처리했다. 위에서는 한줄 한줄 새 문자열-획수 페어를 넣어 코드가 길어졌지만, 아래에서는 match-case를 통해서 길이를 훅 단축시킬 수 있었다. 또 위에서는 뽑은 획수 벡터를 리스트화시켜서 sum을 구했었는데 아래에서는 바로 sum을 구할 수 있어 더 간단해보인다.

## 리스트 버퍼 -> Zip & Flatmap

전 버전에서는 이름 2개를 엇갈리게 겹치기 위해서 빈 리스트 버퍼를 만든 다음, 하나씩 더하는 식으로 작업했다.

{% highlight scala %}

var num_seq = new ListBuffer[Int]()
    num_seq += res1(0)
    num_seq += res2(0)
    num_seq += res1(1)
    num_seq += res2(1)
    num_seq += res1(2)
    num_seq += res2(2)

    // 만든 리스트 버퍼를 리스트로 변환합니다.
    val final_seq = num_seq.toList

{% endhighlight%}

개인적인 습관이 묻어나오는 부분이다. 어떤 리스트나 데이터컬럼이 있을때 보통 공 리스트나 벡터를 만들고, 리스트 안에 item을 하나씩 돌면서 연산을 한 후 공 리스트에 더하는 방식을 많이 사용했었다. Scala의 ListBuffer로 그 방식을 비슷하게 구현했는데, 뭔가 Scala스럽지 않은 느낌이다. 

{% highlight scala %}
// 이름의 획수를 각각 계산한다.
val a_res = name1.toList.map(i => stroke_counter(i))
val b_res = name2.toList.map(i => stroke_counter(i))

// 리스트를 엇갈리게 겹쳐서 하나의 리스트로 만든다
val num_seq = (a_res zip b_res).flatMap(t => List(t._1, t._2))

{% endhighlight%}

이렇게 하면 변형가능한 변수를 만들 필요도 없이 한줄이면 처리가 끝난다. zip으로 (A1, B1), (A2, B2), (A3, B3) 형태로 만든 후, flatmap으로 풀어버리면 [A1, B1, A2, B2, A3, B3]로 쉽게 변형이 가능하다.

<hr>

## 수정된 코드
앞에서 수정한 코드를 이어붙이면 다음과 같이 정리할 수 있고, 빌드하면 바로 main안의 코드가 실행되면서 예시 3의 궁합점수를 뱉게 된다. 스칼라에서는 보통 var로 선언되는 mutable variable보다는 immutable variable(val)이 선호된다고 한다. 함수형 언어인 Scala에서는 val로 변형 불가능한 변수를 자주 쓰기 때문에, 변수 값의 변형이 일어나지 않아 안정적인 프로그래밍(동일한 함수라면 항상 같은 값을 리턴하는)이 가능하다고 한다. 아래 코드에서는 var을 한번도 사용하지 않았으니 얼추 맞는 방향으로 수정하지 않았나 자평해본다.

이후에는 이 스칼라 코드를 이름 2개를 받는 오브젝트로 변환한 후에 궁합 점수를 뱉는 Jar 파일로 만들어 어느 스칼라 프로젝트에서든 import가 가능한 형태로 만들 수 있을 것 같다.

{% highlight scala %}

/**
  * Created by junsik.whang on 2016-10-27.
  */
import com.twitter.penguin.korean.util.Hangul

object name_chemistry {
  def main(args: Array[String]) {

    // 획을 받아서 그에 해당하는 획수를 찾아 반환한다.
    def stroke_finder(a_char: Char): Int = {
      val res = a_char match {
        case 'ㅇ' => 1
        case 'ㄱ' | 'ㄴ' | 'ㅅ' => 2
        case 'ㄷ' | 'ㅈ' | 'ㅋ' | 'ㅎ' => 3
        case 'ㅁ' | 'ㅂ' | 'ㅊ' | 'ㅌ' | 'ㅍ' | 'ㄲ' | 'ㅆ' => 4
        case 'ㄹ' => 5
        case 'ㄸ' | 'ㅉ' => 6
        case 'ㅃ' => 8

        case 'ㅡ' | 'ㅣ' => 1
        case 'ㅏ' | 'ㅓ' | 'ㅗ' | 'ㅜ' | 'ㅢ' => 2
        case 'ㅑ' | 'ㅕ' | 'ㅛ' | 'ㅠ' | 'ㅐ' | 'ㅔ' | 'ㅚ' | 'ㅟ' => 3
        case 'ㅒ' | 'ㅖ' | 'ㅘ' | 'ㅝ' => 4
        case 'ㅙ' | 'ㅞ' => 5

        case 'ㄳ' => 4
        case 'ㄵ' | 'ㄶ' => 5
        case 'ㅄ' => 6
        case 'ㄺ' | 'ㄽ' => 7
        case 'ㅀ' => 8
        case 'ㄻ' | 'ㄼ' | 'ㄾ' | 'ㄿ' => 9

        case _ => 0
      }
      return res
    }

    // '황' 들어오면
    // 'ㅎ', 'ㅘ', 'ㅇ' 으로 짤른 다음
    // 각각에 대해서 획수 계산해서 더한다음 값을 내보낸다.
    def stroke_counter (a_char:Char) : Int = {
      val parsed_hangul = Hangul.decomposeHangul(a_char)

      val parsed_list = parsed_hangul.onset :: parsed_hangul.vowel :: parsed_hangul.coda :: Nil

      val total_stroke = parsed_list.map{ i => stroke_finder(i) }.sum

      return total_stroke
    }

    // 획수를 담은 리스트가 2개 들어오면
    // 리스트를 엇갈리게 합친 후에 궁합 점수를 계산해서 궁합 점수를 산출한다
    def chemistry_calculator (name1: String, name2: String): Int = {

      val a_res = name1.toList.map(i => stroke_counter(i))
      val b_res = name2.toList.map(i => stroke_counter(i))


      // 리스트를 엇갈리게 겹쳐서 하나의 리스트로 만든다
      val num_seq = (a_res zip b_res).flatMap(t => List(t._1, t._2))

      // 맨 앞부터 2개씩 더한 값의 1의 자리를 취하여
      // 다음으로 넘기고 남은 숫자가 2개가 될때까지 동일한 연산을 수행한다.
      // 연산이 종료되면 정수 2개를 가진 리스트를 반환한다.
      def chemistry_main_function (sequence: List[Int]): List[Int] = {

        val result = sequence.sliding(2).map(_.sum).toList
        val result2 = result.map( i => i%10 )
        if (result2.length == 2) {
          return result2
        } else {
          chemistry_main_function(result2)
        }
      }

      // 정수 2개를 조합하여 최종 궁합 점수를 산출하여 반환한다.
      val final_result_list = chemistry_main_function(num_seq)
      val final_result = final_result_list(0) * 10 + final_result_list(1)
      return final_result
    }

    // test case
    val a_name = "박근혜"
    val b_name = "최순실"
    val chem_result = chemistry_calculator(a_name, b_name)
    println(s"궁합결과($a_name - $b_name): $chem_result")
  }
}

{% endhighlight%}