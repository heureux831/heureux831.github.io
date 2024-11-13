---
title: 一个小小的spring boot入门
description: 还得是java，让我们来从头认识一下springboot吧，从头开始了解spring boot以及相关内容。
date: 2024-11-05
tags:
  - java
image: ./2024.png
authors:
  - Duffy
---
## 为什么使用Spring？

传统的javaweb开发为什么不被大家广泛使用？而大家偏向于使用Spring等框架。

当然可以，让我们通过代码示例来比较传统的Java Web开发和使用Spring框架开发的区别。

### 传统Java Web开发（Servlet + JSP）

在传统的Java Web开发中，通常使用Servlet来处理HTTP请求，并使用JSP作为视图技术。以下是一个简单的示例：

**Servlet示例（MyServlet.java）:**
```java
import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;

public class MyServlet extends HttpServlet {
    public void doGet(HttpServletRequest request, HttpServletResponse response) 
    throws ServletException, IOException {
        response.setContentType("text/html");
        PrintWriter out = response.getWriter();
        out.println("<html><body>");
        out.println("<h1>Hello, World!</h1>");
        out.println("</body></html>");
    }
}
```

**web.xml配置：**
```xml
<web-app>
    <servlet>
        <servlet-name>MyServlet</servlet-name>
        <servlet-class>MyServlet</servlet-class>
    </servlet>
    <servlet-mapping>
        <servlet-name>MyServlet</servlet-name>
        <url-pattern>/hello</url-pattern>
    </servlet-mapping>
</web-app>
```

在这个例子中，你需要编写Servlet代码来处理请求，并在`web.xml`中配置Servlet映射。这种方式的缺点是配置繁琐，代码与配置耦合度高，不易于测试和维护。

### 使用Spring框架开发

使用Spring框架，我们可以利用Spring MVC来简化Web层的开发。以下是使用Spring MVC的示例：

**Spring MVC Controller示例（MyController.java）:**
```java
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class MyController {

    @GetMapping("/hello")
    public String sayHello() {
        return "hello"; // 返回视图名称
    }
}
```

**Spring配置文件（spring-servlet.xml）:**
```xml
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:context="http://www.springframework.org/schema/context"
       xsi:schemaLocation="http://www.springframework.org/schema/beans
       http://www.springframework.org/schema/beans/spring-beans.xsd
       http://www.springframework.org/schema/context 
       http://www.springframework.org/schema/context/spring-context.xsd">

    <context:component-scan base-package="com.example" />

    <!-- 配置视图解析器 -->
    <bean class="org.springframework.web.servlet.view.InternalResourceViewResolver">
        <property name="prefix" value="/WEB-INF/views/" />
        <property name="suffix" value=".jsp" />
    </bean>
</beans>
```

**JSP视图文件（hello.jsp）:**
```jsp
<html>
<body>
    <h1>Hello, World!</h1>
</body>
</html>
```

在这个Spring MVC的例子中，我们定义了一个控制器`MyController`，其中包含一个处理`/hello`路径请求的方法。Spring会自动将请求映射到对应的控制器方法，并返回视图名称。这种方式的优点是配置简洁，代码与配置分离，易于测试和维护。

通过这两个例子的对比，可以看出Spring框架通过提供更简洁的配置和代码结构，使得Web开发更加高效和易于管理。


Spring框架因其众多优势而被广泛使用，以下是一些关键原因：

1. **简化开发和解耦**：Spring通过其控制反转(IoC)容器简化了对象之间的依赖关系管理，降低了代码耦合度，使得开发更加简单。

2. **面向切面编程(AOP)**：Spring支持AOP，允许开发者将日志、安全等横切关注点集中处理，提高了代码的复用性和维护性。

3. **声明式事务管理**：Spring提供声明式事务管理，简化了事务代码的编写，提高了开发效率和质量。

4. **方便程序测试**：Spring支持Junit4等测试框架，使得单元测试变得更加容易和方便。

5. **集成优秀框架**：Spring易于集成各种优秀框架，如Struts、Hibernate、MyBatis等，降低了这些框架的使用难度。

6. **轻量级**：Spring是一个轻量级的框架，与EJB容器相比，它需要的资源更少，开销更低。

7. **社区支持和活跃开发者群体**：Spring拥有庞大的社区和活跃的开发者群体，提供了丰富的资源和文档支持。

8. **扩展性和可扩展性**：Spring设计使其易于扩展和定制，以满足特定功能或集成第三方库的需求。

9. **企业级支持**：许多企业和组织使用Spring进行关键任务应用程序的开发，Spring的稳定性和可靠性得到了广泛认可。

10. **集成能力**：Spring与许多其他技术和框架集成良好，如Hibernate、JPA、Thymeleaf等，提高了开发效率。

11. **与云原生技术的集成**：随着云原生技术的兴起，Spring提供了与容器化、微服务等云原生技术的集成方案。

12. **非侵入式设计**：Spring是一种非侵入式框架，使得应用程序代码对框架的依赖最小化。

13. **支持MVC架构**：Spring提供了基于MVC的Web框架，是Struts等其他Web框架的良好替代品。

## 提出Spring

Spring框架因其上述优势而被广泛采用，特别是在企业级Java应用开发中。它通过提供一站式的解决方案，支持广泛的应用场景，从传统的企业应用到现代的云原生应用，Spring都能提供强大的支持。Spring的灵活性、易用性和社区支持使其成为Java开发者的首选框架之一。



## Spring boot?是新框架吗?


Spring boot 是一个基于Spring的框架，主要是为了简化Spring应用的配置和开发过程，让开发者能够快速搭建独立、生产级的应用程序。


首先使用gradle构建项目（管理项目使用的包等）



### build.gradle



## Spring

## 四、Bean 规范

- 所有属性为 private
- 提供默认构造方法
- 提供 getter 和 setter
- 实现 Serializable （比如可以实现Serializable 接口，用于实现bean的持久性）
- 属性类型使用包装类