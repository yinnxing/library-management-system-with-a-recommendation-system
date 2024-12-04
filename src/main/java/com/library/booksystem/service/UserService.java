package com.library.booksystem.service;

import com.library.booksystem.dto.request.UserRequest;
import com.library.booksystem.dto.response.UserResponse;
import com.library.booksystem.enums.Role;
import com.library.booksystem.exception.AppException;
import com.library.booksystem.exception.ErrorCode;
import com.library.booksystem.mapper.UserMapper;
import com.library.booksystem.model.User;
import com.library.booksystem.repository.UserRepository;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
@RequiredArgsConstructor
public class UserService {
    UserRepository userRepository;
    UserMapper userMapper;
    PasswordEncoder passwordEncoder;


    public UserResponse createUser(UserRequest request) {
        if(userRepository.existsByUsername(request.getUsername())){
            throw new AppException(ErrorCode.USER_EXISTED);
        }
        User user = userMapper.toUser(request);
        user.setPassword(passwordEncoder.encode(request.getPassword()));
        user.setRole(Role.USER.name());
        userRepository.save(user);
        return userMapper.toUserResponse(user);
    }
    public UserResponse getMyInfo() {
        var context = SecurityContextHolder.getContext();
        String email = context.getAuthentication().getName();
        User user = userRepository.findByEmail(email).orElseThrow(
                () -> new AppException(ErrorCode.USER_NOT_EXIST)
        );
        return userMapper.toUserResponse(user);

    }
    public List<UserResponse> getUsers(){
        List<User> users = userRepository.findAll();
        return users.stream().map(userMapper::toUserResponse)
                .collect(Collectors.toList());
    }







}
