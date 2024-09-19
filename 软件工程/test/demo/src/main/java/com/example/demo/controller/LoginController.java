package com.example.demo.controller;

import com.example.demo.model.Post;
import com.example.demo.model.User;
import com.example.demo.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.multipart.MultipartFile;
import com.example.demo.service.PostService;
import javax.servlet.http.HttpSession;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

@Controller
public class LoginController {

    @Autowired
    private UserService userService;

    @Autowired
    private PostService postService;

    @GetMapping("/login")
    public String login() {
        return "login";
    }

    @PostMapping("/login")
    public String loginSubmit(@RequestParam String uname,
                              @RequestParam String psw,
                              HttpSession session,
                              Model model) {
        User user = userService.findUserByNameAndPassword(uname, psw);
        if (user != null) {
            session.setAttribute("user", user);
            return "redirect:/userProfile";
        } else {
            model.addAttribute("error", "用户名或密码不正确，请重试。");
            return "login";
        }
    }

    @GetMapping("/userProfile")
    public String userProfile(Model model, HttpSession session) {
        User user = (User) session.getAttribute("user");
        List<Post> posts = postService.getAllPosts();
        model.addAttribute("posts", posts);
        if (user != null) {
            model.addAttribute("user", user);
            return "userProfile";
        } else {
            return "redirect:/login";
        }
    }

    @PostMapping("/uploadAvatar")
    public String uploadAvatar(@RequestPart("avatar") MultipartFile avatar,
                               HttpSession session) throws IOException {
        User user = (User) session.getAttribute("user");
        if (user != null && !avatar.isEmpty()) {
            String filename = user.getUid() + ".jpg";
            String uploadDir = ("D:\\d_code\\git\\软件工程\\test\\demo\\src\\main\\resources\\static\\avatar").toString();
            File file = new File(uploadDir, filename);
            avatar.transferTo(file);
            user.setAvatar(filename);
            userService.updateUser(user); // Ensure this method exists to update user info
        }
        return "redirect:/userProfile";
    }
}
